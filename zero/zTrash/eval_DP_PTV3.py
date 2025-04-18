from zero.expForwardKinematics.ObsProcessor.ObsProcessorPtv3_fk import ObsProcessorPtv3
from zero.expForwardKinematics.trainer_DP import TrainerDP
import re
from zero.expForwardKinematics.config.default import build_args
import yaml
from typing import Tuple, Dict, List
import einops
import os
import json
import jsonlines
import tap
import copy
from pathlib import Path
from filelock import FileLock
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
import torch
import numpy as np
from scipy.special import softmax
from pyrep.errors import IKError, ConfigurationPathError

import pickle
from zero.env.rlbench_lotus.environments import RLBenchEnv, Mover
from zero.expForwardKinematics.config.default import get_config
from zero.expForwardKinematics.config.constants import get_robot_workspace, get_rlbench_labels
import random
from zero.env.rlbench_lotus.recorder import (
    TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion
)
from rlbench.backend.exceptions import InvalidActionError
import torch.multiprocessing as mp
from termcolor import colored

from zero.z_utils.joint_position import denormalize_JP
# ----------------------------------------------
# region utils


def task_file_to_task_class(task_file):
    import importlib
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_to_file(filepath, data):
    lock = FileLock(filepath + '.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)


def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or nparray int, shape=(N, )
    Returns:
        masks: nparray, shape=(N, L), padded=0
    """
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# endregion

# ----------------------------------------------
# region Actioner


class Actioner(object):
    def __init__(self, args) -> None:
        self.args = args
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)

        self.WORKSPACE = get_robot_workspace(real_robot=args.real_robot)
        self.device = torch.device(args.device)

        # config = get_config(args.model_config_path, args.remained_args)
        with open(args.model_config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.config = config['config']

        model = TrainerDP.load_from_checkpoint(checkpoint_path=args.checkpoint, config=self.config)
        self.model = model.policy
        self.model.to(self.device)
        self.model.eval()

        self.obs_processor = ObsProcessorPtv3(self.config, train_flag=False)
        self.collect_fn = self.obs_processor.get_collect_function()

    def preprocess_obs(self, taskvar, step_id, obs):
        # with open('/data/zero/1_Data/C_Dataset_Example/example_obs.pkl', 'wb') as f:
        #     pickle.dump(obs, f)
        # raise NotImplementedError
        obs_raw = self.obs_processor.obs_2_obs_raw(obs)
        obs_static = self.obs_processor.static_process_DA3D(obs_raw)
        obs_dynamic = self.obs_processor.dynamic_process(obs_static, taskvar)
        batch = self.collect_fn(obs_dynamic)
        for item in batch:
            if isinstance(batch[item], torch.Tensor):
                batch[item] = batch[item].to(self.device)
        return batch

    def predict(
        self, task_str=None, variation=None, step_id=None, obs_state_dict=None,
        episode_id=None, instructions=None,
    ):

        # print(obs_state_dict)
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(taskvar, step_id, obs_state_dict,)
        with torch.no_grad():
            actions = self.model.inference_one_sample_JP(batch)[0].data.cpu()  # 原本这里是(7) # 现在，这里要变成(horizon_length,7)
            # actions analysis
            if type(actions) == list:
                actions = torch.stack(actions, 0)
            if len(actions.shape) == 1:
                # single horizon
                actions = actions.unsqueeze(0)
            if len(actions.shape) == 3:
                actions = actions.squeeze(0)
            # check actions shape
            assert len(actions.shape) == 2
            assert actions.shape[1] == 8

        # sigmoid
        actions = einops.rearrange(actions, 'a h -> h a')  # theta_positions, horizon --> horizon, theta_positions
        new_actions = []
        for i, action in enumerate(actions):
            action = denormalize_JP(action)
            new_actions.append(action)
        actions = np.stack(new_actions, 0)

        out = {
            'actions': actions
        }

        if self.args.save_obs_outs_dir is not None:
            np.save(
                os.path.join(self.args.save_obs_outs_dir, f'{task_str}+{variation}-{episode_id}-{step_id}.npy'),
                {
                    'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    'obs': obs_state_dict,
                    'action': action
                }
            )
        return out

# endregion
# ----------------------------------------------
# region MultiProcess

# region consumer


def consumer_fn(args, batch_queue, result_queues):
    print('consumer start')
    # build model
    set_random_seed(args.seed)
    actioner = Actioner(args)

    while True:
        data = batch_queue.get()
        if data is None:
            print('Received None value -> Producers finished.')
            break

        # run one batch
        k_prod, batch = data
        out = actioner.predict(**batch)
        result_queues[k_prod].put(out)
# endregion
# region producer


def producer_fn(proc_id, k_res, args, taskvar, pred_file, batch_queue, result_queue, producer_queue):
    task_str, variation = taskvar.split('+')
    variation = int(variation)

    set_random_seed(args.seed)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        if not os.path.exists(str(episodes_dir)):
            print(f'{taskvar} does not need to be evaluated.')
            producer_queue.put((proc_id, k_res))
            return

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=0,
        action_mode='theta_position',
    )

    env.env.launch()
    task_type = task_file_to_task_class(task_str)
    task = env.env.get_task(task_type)
    task.set_variation(variation)  # type: ignore

    if args.record_video:
        # Add a global camera to the scene
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam_resolution = [args.video_resolution, args.video_resolution]
        cam = VisionSensor.create(cam_resolution)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)

        if args.video_rotate_cam:
            global_cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
        else:
            global_cam_motion = StaticCameraMotion(cam)

        cams_motion = {"global": global_cam_motion}

        if not args.not_include_robot_cameras:
            # Env cameras
            cam_left = VisionSensor.create(cam_resolution)
            cam_right = VisionSensor.create(cam_resolution)
            cam_wrist = VisionSensor.create(cam_resolution)

            left_cam_motion = AttachedCameraMotion(cam_left, task._scene._cam_over_shoulder_left)
            right_cam_motion = AttachedCameraMotion(cam_right, task._scene._cam_over_shoulder_right)
            wrist_cam_motion = AttachedCameraMotion(cam_wrist, task._scene._cam_wrist)

            cams_motion["left"] = left_cam_motion
            cams_motion["right"] = right_cam_motion
            cams_motion["wrist"] = wrist_cam_motion
        tr = TaskRecorder(cams_motion, fps=30)
        task._scene.register_step_callback(tr.take_snap)

        video_log_dir = os.path.join(args.video_dir, f'{task_str}+{variation}')
        os.makedirs(str(video_log_dir), exist_ok=True)

    move = Mover(task, max_tries=args.max_tries)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        demos = []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                try:
                    demo = env.get_demo(task_str, variation, idx, load_images=False)
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
        if len(demos) == 0:
            print(f'{taskvar} does not need to be evaluated.')
            return
    else:
        demos = None

    num_demos = len(demos) if demos is not None else args.num_demos

    success_rate = 0.0
    for demo_id in range(num_demos):
        reward = None

        if demos is None:
            instructions, obs = task.reset()
        else:
            instructions, obs = task.reset_to_demo(demos[demo_id])

        obs_state_dict = env.get_observation(obs)  # type: ignore
        move.reset(obs_state_dict['gripper'])

        for step_id in range(args.max_steps):
            # fetch the current observation, and predict one action
            batch = {
                'task_str': task_str,
                'variation': variation,
                'step_id': step_id,
                'obs_state_dict': obs,
                'episode_id': demo_id,
                'instructions': instructions,
            }
            batch_queue.put((k_res, batch))

            output = result_queue.get()
            actions = output["actions"]

            if actions is None:
                break

            # update the observation based on the predicted action
            for action in actions:
                try:
                    obs, reward, terminate, _ = move(action, verbose=False)
                    if reward == 1:
                        success_rate += 1 / num_demos
                        break
                    if terminate:
                        print("The episode has terminated!")
                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(taskvar, demo_id, step_id, e)
                    reward = 0
                    break

            if reward == 1:
                break

        if args.record_video:  # and reward < 1:
            tr.save(os.path.join(video_log_dir, f"{demo_id}_SR{reward}"))

        print(
            taskvar, "Demo", demo_id, 'Step', step_id + 1,
            "Reward", reward, "Accumulated SR: %.2f" % (success_rate * 100),
            'Estimated SR: %.2f' % (success_rate * num_demos / (demo_id + 1) * 100)
        )

    write_to_file(
        pred_file,
        {
            'checkpoint': args.checkpoint,
            'task': task_str, 'variation': variation,
            'num_demos': num_demos, 'sr': success_rate
        }
    )

    env.env.shutdown()
    print(colored(f'Taskvar: {taskvar} SR: {success_rate:.2f}', 'black', 'on_yellow'))
    producer_queue.put((proc_id, k_res))
# endregion
# region main


def main():
    '''
    get expriment folder which should be the version folder of pytorch lightning
    args:
    exp-config: the path of the eval config file

    '''
    mp.set_start_method('spawn')
    eval_config = build_args('/data/zero/zero/expForwardKinematics/config/eval.yaml')
    eval_config.defrost()

    exp_dir = eval_config.exp_dir

    # choose ckpt
    ckpt_path_all = sorted(os.listdir(os.path.join(exp_dir, 'checkpoints')), key=natural_sort_key)
    if eval_config.epoch is not None:
        for i, ckpt_path in enumerate(ckpt_path_all):
            if f'epoch={eval_config.epoch}' in ckpt_path:
                ckpt_path = os.path.join(exp_dir, 'checkpoints', ckpt_path)
                ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
                break
    else:
        ckpt_path = os.path.join(exp_dir, 'checkpoints', ckpt_path_all[-1])
        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]

    # get model config

    model_config_path = os.path.join(exp_dir, 'hparams.yaml')
    eval_config.model_config_path = model_config_path
    # for ckpt_num in check_point_number:

    ckpt_name = ckpt_path.split('/')[-1].split('.')[0]

    eval_config.expr_dir = f'/data/zero/3_Eval/eval_log/{ckpt_name}/preds'
    eval_config.video_dir = f'/data/zero/3_Eval/videos/{ckpt_name}/videos'
    eval_config.checkpoint = ckpt_path

    if not os.path.exists(eval_config.checkpoint):
        print(eval_config.checkpoint, 'not exists')
        return

    pred_dir = os.path.join(eval_config.expr_dir, f'seed{eval_config.seed}')
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, 'results.jsonl')
    existed_taskvars = set()
    if os.path.exists(pred_file):
        with jsonlines.open(pred_file, 'r') as f:
            for item in f:
                item_step = int(os.path.basename(item['checkpoint']).split('.')[0].split('_')[-1])
                if item_step == eval_config.ckpt_step:
                    existed_taskvars.add(f"{item['task']}+{item['variation']}")

    taskvars = json.load(open(eval_config.taskvar_file))
    taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
    print('checkpoint', eval_config.ckpt_step, '#taskvars', len(taskvars))

    # taskvar_to_use
    if eval_config.tasks_to_use is not None:
        taskvars = [taskvar for taskvar in taskvars if taskvar.split('_peract')[0] in eval_config.tasks_to_use]

    batch_queue = mp.Queue(eval_config.queue_size)
    result_queues = [mp.Queue(eval_config.queue_size) for _ in range(eval_config.num_workers)]
    producer_queue = mp.Queue(eval_config.queue_size)

    consumer = mp.Process(target=consumer_fn, args=(eval_config, batch_queue, result_queues))
    consumer.start()

    producers = {}
    i, k_res = 0, 0
    while i < len(taskvars):
        taskvar = taskvars[i]
        if len(producers) < eval_config.num_workers:
            print('start', i, taskvar)
            producer = mp.Process(
                target=producer_fn,
                args=(i, k_res, eval_config, taskvar, pred_file, batch_queue, result_queues[k_res], producer_queue),
                name=taskvar
            )
            producer.start()
            producers[i] = producer
            i += 1
            k_res += 1
        else:
            proc_id, k_res = producer_queue.get()
            producers[proc_id].join()
            del producers[proc_id]
            # producers[0].join()
            # producers = producers[1:]

    for p in producers.values():
        p.join()

    batch_queue.put(None)
    consumer.join()
    # endregion
# endregion


if __name__ == '__main__':

    main()
