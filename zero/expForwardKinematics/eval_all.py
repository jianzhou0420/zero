'''
Single Process Evaluator
'''
# utils
import re
import yaml
import einops
import os
import json
import jsonlines
import torch
import numpy as np
from filelock import FileLock
import random
import torch.multiprocessing as mp
from termcolor import colored
from typing import Callable
from copy import deepcopy as copy
from numpy import array as npa

# rlbench
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from termcolor import cprint
# homemade rlbench
from zero.env.rlbench_lotus.environments import RLBenchEnv, Mover
from zero.env.rlbench_lotus.recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion

# policy & pytorch-lightning
from zero.expForwardKinematics.config.default import build_args
from zero.expForwardKinematics.ObsProcessor.ObsProcessorFKAll import *
from zero.expForwardKinematics.models.Base.BaseAll import BasePolicy
from zero.expForwardKinematics.trainer_FK_all import Trainer_all, OBS_FACTORY
# homemade utils


# ----------------------------------------------
# region Actioner
class Actioner(object):
    '''
    接受obs,返回操作,每次接受的obs都应该是完整history
    '''

    def __init__(self, eval_config) -> None:
        self.args = eval_config
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)
        self.device = torch.device(eval_config['device'])
        with open(eval_config.model_config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.config = config['config']

        model_name = self.config['Trainer']['model_name']
        policy = Trainer_all.load_from_checkpoint(eval_config['checkpoint'], config=self.config)

        self.model = policy.policy  # type: BasePolicy

        self.obs_processor = OBS_FACTORY[model_name](self.config, train_flag=False)  # type: ObsProcessorRLBenchBase
        self.obs_processor.dataset_init()

        self.model_name = model_name

    def preprocess_obs(self, taskvar, step_id, obs: list[Observation]):

        obs_raw = self.obs_processor.obs_2_obs_raw(obs)
        obs_static = self.obs_processor.static_process(obs_raw)
        obs_dynamic = self.obs_processor.dynamic_process(obs_static, taskvar)
        batch = self.obs_processor.collate_fn([obs_dynamic])

        for item in batch:
            if isinstance(batch[item], torch.Tensor):
                batch[item] = batch[item].to(self.device)
        return batch

    def process_and_save_actions(self):
        pass

    def forward(self, task_str=None, variation=None, step_id=None, obs=None, episode_id=None, instructions=None,):
        # print(obs_state_dict)
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(taskvar, step_id, obs,)
        actions = self.model.inference_one_sample(batch)  # assume it has shape(1,H,action_dim)
        new_actions = self.obs_processor.denormalize_action(actions)

        out = {
            'actions': new_actions[1:2],
        }

        if self.args.save_obs_outs_dir is not None:
            np.save(
                os.path.join(self.args.save_obs_outs_dir, f'{task_str}+{variation}-{episode_id}-{step_id}.npy'),
                {
                    'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    'obs': obs,
                    'eePose': new_actions
                }
            )
        return out


class Evaluator(object):
    def __init__(self):
        # 1. get eval config and train config
        eval_config = build_args('./zero/expForwardKinematics/config/eval_all.yaml')
        eval_config.defrost()
        exp_dir = eval_config['exp_dir']
        model_config_path = os.path.join(exp_dir, 'hparams.yaml')
        eval_config['model_config_path'] = model_config_path
        with open(model_config_path, "r") as f:   # extra code to get action_mode
            tmp = yaml.load(f, Loader=yaml.UnsafeLoader)
        action_mode = tmp['config']['DP']['ActionHead']['action_mode']
        eval_config['action_mode'] = action_mode
        del tmp, action_mode

        # 2. choose ckpt
        ckpt_path_all = sorted(os.listdir(os.path.join(exp_dir, 'checkpoints')), key=natural_sort_key)
        if eval_config['epoch'] is not None:
            for i, ckpt_path in enumerate(ckpt_path_all):
                if f"epoch={eval_config['epoch']}" in ckpt_path:
                    ckpt_path = os.path.join(exp_dir, 'checkpoints', ckpt_path)
                    break
        else:
            ckpt_path = os.path.join(exp_dir, 'checkpoints', ckpt_path_all[-1])
        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        cprint(f'ckpt_path: {ckpt_path}', 'blue')

        # 3. define path
        eval_config['expr_dir'] = f'./3_Eval/eval_log/{ckpt_name}/preds'
        eval_config['video_dir'] = f'./3_Eval/videos/{ckpt_name}/videos'
        eval_config['checkpoint'] = ckpt_path
        if not os.path.exists(eval_config['checkpoint']):
            raise FileNotFoundError(f"Checkpoint {eval_config['checkpoint']} does not exist. Please check the path.")

        self.actioner = Actioner(eval_config)
        self.eval_config = eval_config

        self.obs_recorder = []
        self.num_obs_hist = 2

    def run_single_process(self):
        eval_config = self.eval_config
        # 4. define results dir
        pred_dir = os.path.join(eval_config['expr_dir'], f"seed{eval_config['seed']}")
        os.makedirs(pred_dir, exist_ok=True)
        self.pred_file = os.path.join(pred_dir, 'results.jsonl')
        existed_taskvars = set()
        if os.path.exists(self.pred_file):
            with jsonlines.open(self.pred_file, 'r') as f:
                for item in f:
                    item_step = int(os.path.basename(item['checkpoint']).split('.')[0].split('_')[-1])
                    if item_step == eval_config['ckpt_step']:
                        existed_taskvars.add(f"{item['task']}+{item['variation']}")

        # 4. taskvars
        taskvars = json.load(open(eval_config['taskvar_file']))
        taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
        print('checkpoint', eval_config['ckpt_step'], '#taskvars', len(taskvars))
        # taskvar_to_use
        if eval_config['tasks_to_use'] is not None:
            taskvars = [taskvar for taskvar in taskvars if taskvar.split('_peract')[0] in eval_config['tasks_to_use']]
            if eval_config['variations_to_use'] is not None:
                taskvars = [taskvar for taskvar in taskvars if int(taskvar.split('+')[1]) in eval_config['variations_to_use']]

        for taskvar in taskvars:
            self.eval_single_taskvar(taskvar)

    def eval_single_taskvar(self, taskvar):
        # TODO:microstep_data_dir
        task_str, variation = taskvar.split('+')
        variation = int(variation)
        set_random_seed(self.eval_config['seed'])
        if self.eval_config['action_mode'] == 'eePose':
            action_mode = 'eePose'
        elif self.eval_config['action_mode'] == 'JP':
            action_mode = 'JP'

        env = RLBenchEnv(
            data_path='',
            apply_cameras=("left_shoulder", "right_shoulder", "overhead", "wrist", "front"),
            apply_rgb=True,
            apply_pc=True,
            apply_mask=True,
            headless=self.eval_config['headless'],
            image_size=self.eval_config['image_size'],
            cam_rand_factor=0,
            action_mode=action_mode,
        )

        env.env.launch()
        task = env.env.get_task(task_file_to_task_class(task_str))
        task.set_variation(variation)

        if self.eval_config['record_video']:
            # Add a global camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_resolution = [self.eval_config['video_resolution'], self.eval_config['video_resolution']]
            cam = VisionSensor.create(cam_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            if self.eval_config['video_rotate_cam']:
                global_cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), 0.005)
            else:
                global_cam_motion = StaticCameraMotion(cam)

            cams_motion = {"global": global_cam_motion}

            if not self.eval_config['not_include_robot_cameras']:
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

            video_log_dir = os.path.join(self.eval_config['video_dir'], f'{task_str}+{variation}')
            os.makedirs(str(video_log_dir), exist_ok=True)

        move = Mover(task, disabled=True, max_tries=self.eval_config['max_tries'])
        num_demos = self.eval_config['num_demos']

        # main loop
        success_rate = 0.0
        for demo_id in range(num_demos):
            reward = None
            instructions, obs = task.reset()
            self.update_obs_recorder(obs)

            # TODO: refine it
            obs_state_dict = env.get_observation(obs)  # type: ignore
            move.reset(obs_state_dict['gripper'])

            for step_id in range(self.eval_config['max_steps']):
                print("step_id", step_id)
                # fetch the current observation, and predict one action
                batch = {
                    'task_str': task_str,
                    'variation': variation,
                    'step_id': step_id,
                    'obs': self.obs_recorder,
                    'episode_id': demo_id,
                    'instructions': instructions,
                }
                output = self.actioner.forward(**batch)

                actions = output["actions"]

                if actions is None:
                    raise ValueError(f"actions is None, taskvar: {taskvar}, demo_id: {demo_id}, step_id: {step_id}")

                # update the observation based on the predicted action
                for action in actions:
                    try:
                        obs, reward, terminate, _ = move(action, verbose=False)
                        self.update_obs_recorder(obs)
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

            if self.eval_config['record_video']:  # and reward < 1:
                tr.save(os.path.join(video_log_dir, f"{demo_id}_SR{reward}"))

            print(
                taskvar, "Demo", demo_id, 'Step', step_id + 1,
                "Reward", reward, "Accumulated SR: %.2f" % (success_rate * 100),
                'Estimated SR: %.2f' % (success_rate * num_demos / (demo_id + 1) * 100)
            )

        write_to_file(
            self.pred_file,
            {
                'checkpoint': self.eval_config['checkpoint'],
                'task': task_str, 'variation': variation,
                'num_demos': num_demos, 'sr': success_rate
            }
        )

        env.env.shutdown()
        print(colored(f'Taskvar: {taskvar} SR: {success_rate:.2f}', 'black', 'on_yellow'))

    def update_obs_recorder(self, obs):
        # 时刻保持obs的长度是8,注意顺序
        if len(self.obs_recorder) == self.num_obs_hist:
            self.obs_recorder.pop(0)
            self.obs_recorder.append(obs)
        elif len(self.obs_recorder) == 0:
            [self.obs_recorder.append(obs) for _ in range(self.num_obs_hist)]
        elif len(self.obs_recorder) < self.num_obs_hist:
            num_to_add = self.num_obs_hist - len(self.obs_recorder)
            for i in range(num_to_add):
                self.obs_recorder.append(obs)
        else:
            raise ValueError(f"obs_recorder length is {len(self.obs_recorder)}, but it should be 0 or {self.num_obs_hist}.")
# -------------------------------------------------
# region utils


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_to_file(filepath, data):
    lock = FileLock(filepath + '.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)


def task_file_to_task_class(task_file):
    import importlib
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class
# endregion


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.run_single_process()
