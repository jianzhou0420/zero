import cv2
from tqdm import tqdm
import yacs.config
import pickle

import json
from zero.dataprocess.utils import natural_sort_key
import os
import einops
import copy
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
from zero.expBaseV5.models.lotus.utils.robot_box import RobotBox
from scipy.special import softmax
import torch

from zero.expBaseV5.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
import matplotlib.pyplot as plt
import random
from zero.expBaseV5.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
'''



'''


def get_robot_workspace(real_robot=False, use_vlm=False):
    if real_robot:
        # ur5 robotics room
        if use_vlm:
            TABLE_HEIGHT = 0.0  # meters
            X_BBOX = (-0.60, 0.2)        # 0 is the robot base
            Y_BBOX = (-0.54, 0.54)  # 0 is the robot base
            Z_BBOX = (-0.02, 0.75)      # 0 is the table
        else:
            TABLE_HEIGHT = 0.01  # meters
            X_BBOX = (-0.60, 0.2)        # 0 is the robot base
            Y_BBOX = (-0.54, 0.54)  # 0 is the robot base
            Z_BBOX = (0, 0.75)      # 0 is the table

    else:
        # rlbench workspace
        TABLE_HEIGHT = 0.7505  # meters

        X_BBOX = (-0.5, 1.5)    # 0 is the robot base
        Y_BBOX = (-1, 1)        # 0 is the robot base
        Z_BBOX = (0.2, 2)       # 0 is the floor

    return {
        'TABLE_HEIGHT': TABLE_HEIGHT,
        'X_BBOX': X_BBOX,
        'Y_BBOX': Y_BBOX,
        'Z_BBOX': Z_BBOX
    }


def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))


class ObsProcessLotus:
    '''
    Processor that convert raw pcd and rgb to voxelized pcd 

    Three stages Process:
    1. Pointlize: RGBD and Camera parameters (both intrincit and extrinsic) to raw pcd [xyz, rgb]
    2. Downsample: raw pcd [xyz, rgb] to voxelized pcd
    3. Trim: remove robot, table, background

    This class only include stage 2 and 3.
    '''

    def __init__(self, config=None, selfgen=True):
        self.rotation_transform = RotationMatrixTransform()
        self.WORKSPACE = get_robot_workspace(real_robot=False, use_vlm=False)
        self.selfgen = selfgen
        self.config = config

        # print(self.config)
        # print('Config loaded')

    ##########################################
    ########## Public Functions ##############
    ##########################################

    # level 1
    def dataset_preprocess_single_episode_with_path(self, data, actions_all, episode_path):
        ''' 
        receive data from selfgen_one_step, this function process single episode data
        input = {
            'key_frameids': [],
            'rgb': [],  # (T, N, H, W, 3)
            'pc': [],  # (T, N, H, W, 3)
            'action': [],  # (T, A)
            'bbox': [],  # [T of dict]
            'pose': []  # [T of dict]
        }

        outs = {
            'pc_fts': [], 
            'step_ids': [],
            'pc_centroids': [],
            'pc_radius': [], 
            'ee_poses': [],
            'txt_embeds': [], 
            'gt_actions': [],
        }
        '''
        outs = {
            'xyz': [],
            'rgb': [],
            'action_current': [],
            'action_next': [],
            'data_ids': [],
            'arm_links_info': [],
            'actions_path': []
        }

        all_names = episode_path.split('/')
        task_name = all_names[6]
        variation_name = all_names[7].split('variation')[-1]
        episode_name = all_names[9]
        taskvar = f'{task_name}_peract+{variation_name}'
        '''
        1. remove outside workspace
        2. remove table
        3. voxelization
        '''
        gt_rots = self._get_groundtruth_rotations(data['action'][:, 3:7])
        for t in range(len(data['key_frameids']) - 1):  # last frame dont train

            # actions_all process
            action_current = copy.deepcopy(data['action'][t])
            action_next = copy.deepcopy(data['action'][t + 1])
            action_current_frame_id = copy.deepcopy(data['key_frameids'][t])
            action_next_frame_id = copy.deepcopy(data['key_frameids'][t + 1])
            action_path = copy.deepcopy(actions_all[action_current_frame_id:action_next_frame_id + 1])  # 需要包头包尾
            assert (action_path[0] == action_current).all(), f'{action_path[0]} != {action_current}'
            assert (action_path[-1] == action_next).all(), f'{action_path[-1]} != {action_next}'
            outs['actions_path'].append(action_path)

            del action_current, action_next, action_current_frame_id, action_next_frame_id
            # 0.retrieve data
            xyz = data['pc'][t].reshape(-1, 3)
            rgb = data['rgb'][t].reshape(-1, 3)

            arm_links_info = (data['bbox'][t], data['pose'][t])
            action_current = copy.deepcopy(data['action'][t])
            action_next = copy.deepcopy(data['action'][t + 1])

            gt_rot = gt_rots[t]

            # 1.remove ouside workspace
            in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                      (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                      (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])

            # 2. remove table
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
            xyz = xyz[in_mask]
            rgb = rgb[in_mask]

            # 4. remove robot
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.config.TRAIN_DATASET.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

            # 3. voxelization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd, _, trace = pcd.voxel_down_sample_and_trace(
                self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
            )
            xyz = np.asarray(pcd.points)
            trace = np.array([v[0] for v in trace])
            rgb = rgb[trace]

            xyz = np.array(xyz, dtype=np.float32)
            rgb = np.array(rgb, dtype=np.float32)
            action_current = np.array(action_current, dtype=np.float32)
            action_next = np.array(action_next, dtype=np.float32)
            # arm_links_info = np.array(arm_links_info, dtype=np.float32)

            outs['xyz'].append(xyz)
            outs['rgb'].append(rgb)
            outs['action_current'].append(action_current)
            outs['action_next'].append(action_next)
            outs['data_ids'].append(f'{task_name}-{variation_name}-{episode_name}-t{t}')
            outs['arm_links_info'].append(arm_links_info)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd.colors = o3d.utility.Vector3dVector((rgb + 1) / 2)

            # o3d.visualization.draw_geometries([pcd])
        return outs

    def dataset_preprocess_single_episode(self, data, episode_path):
        ''' 
        receive data from selfgen_one_step, this function process single episode data
        input = {
            'key_frameids': [],
            'rgb': [],  # (T, N, H, W, 3)
            'pc': [],  # (T, N, H, W, 3)
            'action': [],  # (T, A)
            'bbox': [],  # [T of dict]
            'pose': []  # [T of dict]
        }

      outs = {
            'pc_fts': [], 
            'step_ids': [],
            'pc_centroids': [],
            'pc_radius': [], 
            'ee_poses': [],
            'txt_embeds': [], 
            'gt_actions': [],
        }
        '''
        outs = {
            'xyz': [],
            'rgb': [],
            'action_current': [],
            'action_next': [],
            'data_ids': [],
            'arm_links_info': []
        }

        all_names = episode_path.split('/')
        task_name = all_names[6]
        variation_name = all_names[7].split('variation')[-1]
        episode_name = all_names[9]
        taskvar = f'{task_name}_peract+{variation_name}'
        '''
        1. remove outside workspace
        2. remove table
        3. voxelization
        '''
        gt_rots = self._get_groundtruth_rotations(data['action'][:, 3:7])
        for t in range(len(data['key_frameids']) - 1):  # last frame dont train
            # 0.retrieve data
            xyz = data['pc'][t].reshape(-1, 3)
            rgb = data['rgb'][t].reshape(-1, 3)

            arm_links_info = (data['bbox'][t], data['pose'][t])
            action_current = copy.deepcopy(data['action'][t])
            action_next = copy.deepcopy(data['action'][t + 1])

            gt_rot = gt_rots[t]

            # 1.remove ouside workspace
            in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                      (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                      (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])

            # 2. remove table
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
            xyz = xyz[in_mask]
            rgb = rgb[in_mask]

            # 4. remove robot
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.config.TRAIN_DATASET.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

            # 3. voxelization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd, _, trace = pcd.voxel_down_sample_and_trace(
                self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
            )
            xyz = np.asarray(pcd.points)
            trace = np.array([v[0] for v in trace])
            rgb = rgb[trace]

            xyz = np.array(xyz, dtype=np.float32)
            rgb = np.array(rgb, dtype=np.float32)
            action_current = np.array(action_current, dtype=np.float32)
            action_next = np.array(action_next, dtype=np.float32)
            # arm_links_info = np.array(arm_links_info, dtype=np.float32)

            outs['xyz'].append(xyz)
            outs['rgb'].append(rgb)
            outs['action_current'].append(action_current)
            outs['action_next'].append(action_next)
            outs['data_ids'].append(f'{task_name}-{variation_name}-{episode_name}-t{t}')
            outs['arm_links_info'].append(arm_links_info)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd.colors = o3d.utility.Vector3dVector((rgb + 1) / 2)

            # o3d.visualization.draw_geometries([pcd])

        return outs

    def dataset_preprocess_single_episode_edge_detection(self, data, episode_path):
        ''' 
        receive data from selfgen_one_step, this function process single episode data
        input = {
            'key_frameids': [],
            'rgb': [],  # (T, N, H, W, 3)
            'pc': [],  # (T, N, H, W, 3)
            'action': [],  # (T, A)
            'bbox': [],  # [T of dict]
            'pose': []  # [T of dict]
        }

      outs = {
            'data_ids': [], 
            'pc_fts': [], 
            'step_ids': [],
            'pc_centroids': [],
            'pc_radius': [], 
            'ee_poses': [],
            'txt_embeds': [], 
            'gt_actions': [],
        }
        '''
        outs = {
            'xyz': [],
            'rgb': [],
            'action_current': [],
            'action_next': [],
            'data_ids': [],
            'arm_links_info': []
        }

        all_names = episode_path.split('/')
        task_name = all_names[6]
        variation_name = all_names[7].split('variation')[-1]
        episode_name = all_names[9]

        '''
        1. remove outside workspace
        2. remove table
        3. voxelization
        '''

        for t in range(len(data['key_frameids']) - 1):  # last frame dont train
            # 0.retrieve data
            pc = data['pc'][t]
            images = data['rgb'][t]
            # plt.subplot(2, 2, 1)
            # plt.imshow(images[0])
            # plt.subplot(2, 2, 2)
            # plt.imshow(images[1])
            # plt.subplot(2, 2, 3)
            # plt.imshow(images[2])
            # plt.subplot(2, 2, 4)
            # plt.imshow(images[3])
            # plt.show()

            arm_links_info = (data['bbox'][t], data['pose'][t])
            action_current = copy.deepcopy(data['action'][t])
            action_next = copy.deepcopy(data['action'][t + 1])

            # 1.edge detection
            idxs = []
            for image in images:
                canny_edges = cv2.Canny(image, 150, 200)
                idxs.append(canny_edges > 0)
            all_points = []
            all_points_rgb = []

            for i, idx in enumerate(idxs):
                single_image_points = pc[i][idx]
                single_image_points_rgb = images[i][idx]
                all_points.append(single_image_points)
                all_points_rgb.append(single_image_points_rgb)

            xyz = np.vstack(all_points)
            rgb = np.vstack(all_points_rgb)

            # 2.remove ouside workspace
            in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                      (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                      (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
            # 3. remove table
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
            xyz = xyz[in_mask]
            rgb = rgb[in_mask]

            xyz = np.array(xyz, dtype=np.float32)
            rgb = np.array(rgb, dtype=np.float32)
            action_current = np.array(action_current, dtype=np.float32)
            action_next = np.array(action_next, dtype=np.float32)
            # arm_links_info = np.array(arm_links_info, dtype=np.float32)

            outs['xyz'].append(xyz)
            outs['rgb'].append(rgb)
            outs['action_current'].append(action_current)
            outs['action_next'].append(action_next)
            outs['data_ids'].append(f'{task_name}-{variation_name}-{episode_name}-t{t}')
            outs['arm_links_info'].append(arm_links_info)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd.colors = o3d.utility.Vector3dVector((rgb / 255))
            # o3d.visualization.draw_geometries([pcd])

        return outs

    # level 2
    def dataset_preprocess_with_path(self, origin_data_root, output_dir, tasks_to_use=None):
        tasks_list = self._retrieve_all_episodes(origin_data_root)  # 304706
        total = sum([len(variation) for task in tasks_list for variation in task])    # nested sum
        pbar = tqdm(total=total)
        for task in tasks_list:
            # test = task[0][0].split('/')
            task_name = task[0][0].split('/')[-5]
            if tasks_to_use is not None and task_name not in tasks_to_use:
                pbar.update(100)
                continue
            for variation in task:
                for episode in variation:
                    data_path = episode
                    sub = data_path.split('/')
                    export_path = os.path.join(output_dir, *sub[-5:])
                    if os.path.exists(export_path):
                        pbar.update(1)
                        continue
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    actions_all_path = os.path.join('/'.join(data_path.split('/')[:-1]), 'actions_all.pkl')
                    with open(actions_all_path, 'rb') as f:
                        actions_all = pickle.load(f)
                    # print(rgb.shape, xyz.shape)
                    new_data = self.dataset_preprocess_single_episode_with_path(data, actions_all, episode)

                    with open(export_path, 'wb') as f:
                        pickle.dump(new_data, f)
                    pbar.update(1)

    def dataset_preprocess(self, origin_data_root, output_dir, tasks_to_use=None):
        tasks_list = self._retrieve_all_episodes(origin_data_root)  # 304706
        total = sum([len(variation) for task in tasks_list for variation in task])    # nested sum
        pbar = tqdm(total=total)
        for task in tasks_list:
            # test = task[0][0].split('/')
            task_name = task[0][0].split('/')[-5]
            if tasks_to_use is not None and task_name not in tasks_to_use:
                pbar.update(100)
                continue
            for variation in task:
                for episode in variation:
                    data_path = episode
                    sub = data_path.split('/')
                    export_path = os.path.join(output_dir, *sub[-5:])
                    if os.path.exists(export_path):
                        pbar.update(1)
                        continue
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    # print(rgb.shape, xyz.shape)
                    new_data = self.dataset_preprocess_single_episod_with_path(data, episode)

                    with open(export_path, 'wb') as f:
                        pickle.dump(new_data, f)
                    pbar.update(1)

    # level 3
    def dataset_preprocess_train_val(self, origin_data_root, output_dir, tasks_to_use=None):
        train_path = os.path.join(origin_data_root, 'train')
        val_path = os.path.join(origin_data_root, 'val')
        train_output_path = os.path.join(output_dir, 'train')
        val_output_path = os.path.join(output_dir, 'val')

        os.makedirs(train_output_path, exist_ok=1)
        os.makedirs(val_output_path, exist_ok=1)
        self.dataset_preprocess(train_path, train_output_path, tasks_to_use)
        self.dataset_preprocess(val_path, val_output_path, tasks_to_use)

    def dataset_preprocess_train_val_with_path(self, origin_data_root, output_dir, tasks_to_use=None):
        train_path = os.path.join(origin_data_root, 'train')
        val_path = os.path.join(origin_data_root, 'val')
        train_output_path = os.path.join(output_dir, 'train')
        val_output_path = os.path.join(output_dir, 'val')

        os.makedirs(train_output_path, exist_ok=1)
        os.makedirs(val_output_path, exist_ok=1)
        self.dataset_preprocess_with_path(train_path, train_output_path, tasks_to_use)
        self.dataset_preprocess_with_path(val_path, val_output_path, tasks_to_use)

    def dataset_preprocess_edge_detection(self, origin_data_root, output_dir, tasks_to_use=None):
        tasks_list = self._retrieve_all_episodes(origin_data_root)
        total = sum([len(variation) for task in tasks_list for variation in task])    # nested sum
        pbar = tqdm(total=total)
        for task in tasks_list:
            # test = task[0][0].split('/')
            task_name = task[0][0].split('/')[6]
            if tasks_to_use is not None and task_name not in tasks_to_use:
                pbar.update(100)
                continue
            for variation in task:
                for episode in variation:
                    data_path = episode
                    sub = data_path.split('/seed42')
                    export_path = os.path.join(output_dir, sub[1][1:])
                    if os.path.exists(export_path):
                        pbar.update(1)
                        continue
                    os.makedirs(os.path.dirname(export_path), exist_ok=True)
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    # print(rgb.shape, xyz.shape)
                    new_data = self.dataset_preprocess_single_episode_edge_detection(data, episode)

                    with open(export_path, 'wb') as f:
                        pickle.dump(new_data, f)
                    pbar.update(1)

    ##########################################
    ########## Private Functions #############
    ##########################################

    def _get_groundtruth_rotations(self, action,):
        gt_rots = torch.from_numpy(action.copy())   # quaternions
        rot_type = self.config.TRAIN_DATASET.rot_type
        euler_resolution = self.config.TRAIN_DATASET.euler_resolution
        if rot_type == 'euler':    # [-1, 1]
            gt_rots = self.rotation_transform.quaternion_to_euler(gt_rots[1:]) / 180.
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        elif rot_type == 'euler_disc':  # 3D
            gt_rots = [quaternion_to_discrete_euler(x, euler_resolution) for x in gt_rots[1:]]
            gt_rots = torch.from_numpy(np.stack(gt_rots + gt_rots[-1:]))
        elif rot_type == 'euler_delta':
            gt_eulers = self.rotation_transform.quaternion_to_euler(gt_rots)
            gt_rots = (gt_eulers[1:] - gt_eulers[:-1]) % 360
            gt_rots[gt_rots > 180] -= 360
            gt_rots = gt_rots / 180.
            gt_rots = torch.cat([gt_rots, torch.zeros(1, 3)], 0)
        elif rot_type == 'rot6d':
            gt_rots = self.rotation_transform.quaternion_to_ortho6d(gt_rots)
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        else:
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        gt_rots = gt_rots.numpy()
        return gt_rots

    def _retrieve_all_episodes(self, root_path):
        tasks_list = []
        all_tasks = sorted(os.listdir(root_path), key=natural_sort_key)
        for task in all_tasks:
            variations_list = []
            tasks_list.append(variations_list)
            task_path = os.path.join(root_path, task)
            all_variations = sorted(os.listdir(task_path), key=natural_sort_key)
            for variation in all_variations:
                episodes_list = []
                variations_list.append(episodes_list)
                single_variation_path = os.path.join(task_path, variation, 'episodes')
                all_episodes = sorted(os.listdir(single_variation_path), key=natural_sort_key)
                for episode in all_episodes:
                    single_episode_path = os.path.join(single_variation_path, episode, 'data.pkl')
                    episodes_list.append(single_episode_path)

        return tasks_list

    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False
        robot_box = RobotBox(
            arm_links_info, keep_gripper=keep_gripper,
            env_name='rlbench', selfgen=True
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask

    def _augment_pc(self, xyz, action_current, action_next, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        action_current[:3] = random_rotate_z(action_current[:3], angle=angle)
        action_current[3:-1] = self._rotate_gripper(action_current[3:-1], angle)

        action_next[:3] = random_rotate_z(action_next[:3], angle=angle)
        action_next[3:-1] = self._rotate_gripper(action_next[3:-1], angle)

        if self.config.TRAIN_DATASET.rot_type == 'quat':
            gt_rot = action_next[3:-1]
        elif self.config.TRAIN_DATASET.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(action_next[3:-1][None, :]))[0].numpy() / 180.
        elif self.config.TRAIN_DATASET.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(action_next[3:-1], self.config.TRAIN_DATASET.euler_resolution)
        elif self.config.TRAIN_DATASET.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(action_next[3:-1][None, :]))[0].numpy()

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, action_current, action_next, gt_rot

    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot

    ##########################################
    ##########     Functions   ###############
    ##########################################

    def inside_workspace(self, xyz, rgb):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        return xyz[in_mask], rgb[in_mask]

    def visualize_pc(self, xyz, rgb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

    def remove_table(self, xyz, rgb):
        in_mask = xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT']
        return xyz[in_mask], rgb[in_mask]

    def remove_robot(self, xyz, rgb):
        mask = self._get_mask_with_robot_box(xyz, self.rm_robot_type)
        return xyz[mask], rgb[mask]


if __name__ == '__main__':
    # mixed args
    from zero.expBaseV5.config.default import build_args
    config = build_args()

    op = ObsProcessLotus(config, selfgen=True)

    op.dataset_preprocess_train_val_with_path(config.A_Selfgen, config.B_Preprocess, tasks_to_use=None)

    '''
    python -m zero.dataprocess.ObsProcessor\
        --exp-config /data/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001\
        MODEL.action_config.pos_bins 75\
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use "insert_onto_square_peg"
     
     # tasks_to_use "[meat_off_grill, sweep_to_dustpan_of_size, close_jar, push_buttons, light_bulb_in, insert_onto_square_peg, put_groceries_in_cupboard,place_shape_in_shape_sorter,stack_blocks]" \
    '''
