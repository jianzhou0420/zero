from typing import Tuple, Dict, List

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
import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from zero.expBins.models.lotus.model_expbase import SimplePolicyPTV3CA

from zero.env.rlbench_lotus.environments import RLBenchEnv, Mover

from zero.expBins.config.default import get_config

from zero.expBins.config.constants import get_robot_workspace, get_rlbench_labels
from zero.z_utils.robot_box import RobotBox
import random
from zero.env.rlbench_lotus.recorder import (
    TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion
)
from rlbench.backend.exceptions import InvalidActionError
import torch.multiprocessing as mp
from termcolor import colored
from zero.expBins.trainer_expbase import TrainerLotus


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


class EvalArgs(tap.Tap):
    expr_dir: str = '/data/ckpt/'

    device: str = 'cuda'  # cpu, cuda

    image_size: List[int] = [512, 512]
    max_tries: int = 3
    max_steps: int = 15

    microstep_data_dir: str = ''
    num_workers: int = 1
    queue_size: int = 20
    taskvar_file: str = '/data/zero/assets/taskvars_peract.json'
    num_ensembles: int = 1

    save_obs_outs_dir: str = None

    best_disc_pos: str = 'max'  # max, ens1

    record_video: bool = True
    video_dir: str = None
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480

    real_robot: bool = False

    ############################
    # sbatch
    ############################
    ckpt_step = 220000
    seed = 42
    num_workers = 4
    num_demos = 10
    config: str = None
    name: str = None
    checkpoint: str = None
    tasks_to_use: List[str] = None
    # microstep_data_dir = '/data/lotus/peract/test/microsteps'


class Actioner(object):
    def __init__(self, args) -> None:
        self.args = args
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)

        self.WORKSPACE = get_robot_workspace(real_robot=args.real_robot)
        self.device = torch.device(args.device)

        config = get_config(args.config, args.remained_args)
        self.config = config
        self.config.defrost()

        if args.checkpoint is not None:
            config.checkpoint = args.checkpoint
        # config.pl_flag=False
        if config.pl_flag:
            model = TrainerLotus.load_from_checkpoint(checkpoint_path=config.checkpoint, config=self.config)
            self.model = model.model
        else:
            self.model = SimplePolicyPTV3CA(config.MODEL)
            if config.checkpoint:
                checkpoint = torch.load(
                    config.checkpoint, map_location=lambda storage, loc: storage
                )
                self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.config.freeze()

        data_cfg = self.config.TRAIN_DATASET
        self.data_cfg = data_cfg
        self.instr_embeds = np.load(data_cfg.instr_embed_file, allow_pickle=True).item()
        if data_cfg.instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        self.taskvar_instrs = json.load(open(data_cfg.taskvar_instr_file))

        self.TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']

    def _get_mask_with_label_ids(self, sem, label_ids):
        mask = sem == label_ids[0]
        for label_id in label_ids[1:]:
            mask = mask | (sem == label_id)
        return mask

    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False
        robot_box = RobotBox(
            arm_links_info, keep_gripper=keep_gripper,
            env_name='real' if self.args.real_robot else 'rlbench'
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask

    def _rm_pc_outliers(self, xyz, rgb=None):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd, idxs = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd, idxs = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
        clf = LocalOutlierFactor(n_neighbors=self.data_cfg.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        return xyz, rgb

    def process_point_clouds(
        self, xyz, rgb, gt_sem=None, ee_pose=None, arm_links_info=None, taskvar=None
    ):
        # 1. In worksapce
        xyz = xyz.reshape(-1, 3)
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                  (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                  (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        # 2. remove table points
        if self.data_cfg.rm_table:
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
        xyz = xyz[in_mask]
        rgb = rgb.reshape(-1, 3)[in_mask]
        if gt_sem is not None:
            gt_sem = gt_sem.reshape(-1)[in_mask]

        # 3. voxelization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        if gt_sem is not None:
            gt_sem = gt_sem[trace]

        if self.args.real_robot:
            for _ in range(1):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
                xyz = xyz[outlier_masks]
                rgb = rgb[outlier_masks]
                if gt_sem is not None:
                    gt_sem = gt_sem[outlier_masks]

        # remove non-object points
        if not self.args.real_robot:
            rm_label_ids = get_rlbench_labels(
                taskvar.split('+')[0], table=self.data_cfg.rm_table, robot=(self.data_cfg.rm_robot == 'gt'), wall=False, floor=False
            )
            if len(rm_label_ids) > 0:
                rm_mask = self._get_mask_with_label_ids(gt_sem, rm_label_ids)
                xyz = xyz[~rm_mask]
                rgb = rgb[~rm_mask]

        if self.data_cfg.rm_robot.startswith('box'):
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.data_cfg.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

        if self.data_cfg.rm_pc_outliers:
            xyz, rgb = self._rm_pc_outliers(xyz, rgb)

        # sampling points
        if len(xyz) > self.data_cfg.num_points:
            if self.data_cfg.sample_points_by_distance:
                dists = np.sqrt(np.sum((xyz - ee_pose[:3])**2, 1))
                probs = 1 / np.maximum(dists, 0.1)
                probs = np.maximum(softmax(probs), 1e-30)
                probs = probs / sum(probs)
                # probs = 1 / dists
                # probs = probs / np.sum(probs)
                point_idxs = np.random.choice(len(xyz), self.data_cfg.num_points, replace=False, p=probs)
            else:
                point_idxs = np.random.choice(len(xyz), self.data_cfg.num_points, replace=False)
        else:
            if self.data_cfg.same_npoints_per_example:
                point_idxs = np.random.choice(xyz.shape[0], self.data_cfg.num_points, replace=True)
            else:
                point_idxs = np.arange(xyz.shape[0])

        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        height = xyz[:, -1] - self.TABLE_HEIGHT

        # normalize
        if self.data_cfg.xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif self.data_cfg.xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif self.data_cfg.xyz_shift == 'gripper':
            centroid = copy.deepcopy(ee_pose[:3])
        if self.data_cfg.xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        ee_pose[:3] = (ee_pose[:3] - centroid) / radius

        rgb = (rgb / 255.) * 2 - 1
        pc_ft = np.concatenate([xyz, rgb], 1)
        if self.data_cfg.get('use_height', False):
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

        return pc_ft, centroid, radius, ee_pose

    def preprocess_obs(self, taskvar, step_id, obs):
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        xyz = np.stack(obs['pc'], 0)  # (N, H, W, C)
        if 'gt_mask' in obs:
            gt_sem = np.stack(obs['gt_mask'], 0)  # (N, H, W)
        else:
            gt_sem = None

        # select one instruction
        instr = self.taskvar_instrs[taskvar][0]
        instr_embed = self.instr_embeds[instr]

        pc_ft, pc_centroid, pc_radius, ee_pose = self.process_point_clouds(
            xyz, rgb, gt_sem=gt_sem, ee_pose=copy.deepcopy(obs['gripper']),
            arm_links_info=obs['arm_links_info'], taskvar=taskvar
        )

        batch = {
            'pc_fts': torch.from_numpy(pc_ft).float(),
            'pc_centroids': pc_centroid,
            'pc_radius': pc_radius,
            'ee_poses': torch.from_numpy(ee_pose).float().unsqueeze(0),
            'step_ids': torch.LongTensor([step_id]),
            'txt_embeds': torch.from_numpy(instr_embed).float(),
            'txt_lens': [instr_embed.shape[0]],
            'npoints_in_batch': [pc_ft.shape[0]],
            'offset': torch.LongTensor([pc_ft.shape[0]]),
        }
        if self.config.MODEL.model_class == 'SimplePolicyPCT':
            batch['pc_fts'] = batch['pc_fts'].unsqueeze(0)
            batch['txt_masks'] = torch.from_numpy(
                gen_seq_masks(batch['txt_lens'])
            ).bool()
            batch['txt_embeds'] = batch['txt_embeds'].unsqueeze(0)

        # for k, v in batch.items():
        #     if k not in ['pc_centroids', 'pc_radius', 'npoints_in_batch']:
        #         print(k, v.size())
        return batch

    def predict(
        self, task_str=None, variation=None, step_id=None, obs_state_dict=None,
        episode_id=None, instructions=None,
    ):
        # print(obs_state_dict)
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(
            taskvar, step_id, obs_state_dict,
        )
        with torch.no_grad():
            actions = []
            # TODO
            for _ in range(self.args.num_ensembles):
                action = self.model(batch)[0].data.cpu()
                actions.append(action)
            if len(actions) > 1:
                # print(torch.stack(actions, 0))
                avg_action = torch.stack(actions, 0).mean(0)
                pred_rot = torch.from_numpy(R.from_euler(
                    'xyz', np.mean([R.from_quat(x[3:-1]).as_euler('xyz') for x in actions], 0),
                ).as_quat())
                action = torch.cat([avg_action[:3], pred_rot, avg_action[-1:]], 0)
            else:
                action = actions[0]
        action[-1] = torch.sigmoid(action[-1]) > 0.5

        # action = action.data.cpu().numpy()
        action = action.numpy()
        # action[:3] = action[:3] * batch['pc_radius'] + batch['pc_centroids']

        # TODO: ensure the action height is above the table
        action[2] = max(action[2], self.TABLE_HEIGHT + 0.005)

        out = {
            'action': action
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
        headless=True,
        image_size=args.image_size,
        cam_rand_factor=0,
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
                'obs_state_dict': obs_state_dict,
                'episode_id': demo_id,
                'instructions': instructions,
            }
            batch_queue.put((k_res, batch))

            output = result_queue.get()
            action = output["action"]

            if action is None:
                break

            # update the observation based on the predicted action
            try:
                obs, reward, terminate, _ = move(action, verbose=False)
                obs_state_dict = env.get_observation(obs)  # type: ignore

                if reward == 1:
                    success_rate += 1 / num_demos
                    break
                if terminate:
                    print("The episode has terminated!")
            except (IKError, ConfigurationPathError, InvalidActionError) as e:
                print(taskvar, demo_id, step_id, e)
                reward = 0
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


def main():
    mp.set_start_method('spawn')
    # for ckpt_num in check_point_number:
    args = EvalArgs().parse_args(known_only=True)
    args.remained_args = args.extra_args
    checkpoint_name = args.checkpoint.split('/')[-1]

    args.expr_dir = f'/data/zero/3_Eval/eval_log/{checkpoint_name}/preds'
    args.video_dir = f'/data/zero/3_Eval/videos/{checkpoint_name}/videos'
    # args.tasks_to_use = ['insert_onto_square_peg']

    if not os.path.exists(args.checkpoint):
        print(args.checkpoint, 'not exists')
        return

    pred_dir = os.path.join(args.expr_dir, f'seed{args.seed}')
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, 'results.jsonl')
    existed_taskvars = set()
    if os.path.exists(pred_file):
        with jsonlines.open(pred_file, 'r') as f:
            for item in f:
                item_step = int(os.path.basename(item['checkpoint']).split('.')[0].split('_')[-1])
                if item_step == args.ckpt_step:
                    existed_taskvars.add(f"{item['task']}+{item['variation']}")

    taskvars = json.load(open(args.taskvar_file))
    taskvars = [taskvar for taskvar in taskvars if taskvar not in existed_taskvars]
    print('checkpoint', args.ckpt_step, '#taskvars', len(taskvars))

    # taskvar_to_use
    if args.tasks_to_use is not None:
        taskvars = [taskvar for taskvar in taskvars if taskvar.split('_peract')[0] in args.tasks_to_use]

    batch_queue = mp.Queue(args.queue_size)
    result_queues = [mp.Queue(args.queue_size) for _ in range(args.num_workers)]
    producer_queue = mp.Queue(args.queue_size)

    consumer = mp.Process(target=consumer_fn, args=(args, batch_queue, result_queues))
    consumer.start()

    producers = {}
    i, k_res = 0, 0
    while i < len(taskvars):
        taskvar = taskvars[i]
        if len(producers) < args.num_workers:
            print('start', i, taskvar)
            producer = mp.Process(
                target=producer_fn,
                args=(i, k_res, args, taskvar, pred_file, batch_queue, result_queues[k_res], producer_queue),
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


if __name__ == '__main__':
    main()
