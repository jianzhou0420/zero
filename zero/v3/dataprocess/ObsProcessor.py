from tqdm import tqdm
import yacs.config
import pickle

import json
from zero.v3.dataprocess.utils import natural_sort_key, get_mask_with_robot_box
import os
import einops
import copy
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import open3d
from zero.v3.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
from zero.v3.models.lotus.utils.robot_box import RobotBox
from scipy.special import softmax
import torch

from zero.v3.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from sklearn.neighbors import LocalOutlierFactor
import random


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


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
        self.rm_pc_outliers_neighbors = 25
        self.rm_robot_type = 'box_keep_gripper'
        self.rotation_transform = RotationMatrixTransform()
        self.WORKSPACE = get_robot_workspace(real_robot=False, use_vlm=False)
        self.selfgen = selfgen
        self.config = config

        if self.config == None:
            default_config_path = '/data/zero/zero/v3/config/after_shock.yaml'
            self.config = yacs.config.CfgNode(new_allowed=True)
            self.config.merge_from_file(default_config_path)

        print(self.config)
        print('Config loaded')

    ##########################################
    ########## Public Functions ##############
    ##########################################

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

    def dataset_preprocess(self, origin_data_root, output_dir, tasks_to_use=None):

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
                    new_data = self.dataset_preprocess_single_episode(data, episode)

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

    ##########################################
    ##########     Functions   ###############
    ##########################################

    def inside_workspace_mask(self, xyz):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        return in_mask


if __name__ == '__main__':
    # mixed args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='exp1_0.005')
    args = parser.parse_args()
    config_path = os.path.join('/data/zero/zero/v3/config', args.config)

    config = yacs.config.CfgNode(new_allowed=True)
    config.merge_from_file(config_path)

    op = ObsProcessLotus(config, selfgen=True)

    # from datetime import datetime

    op.dataset_preprocess(config.selfgen_dir, config.preprocess_dir, tasks_to_use=config.tasks_to_use)
