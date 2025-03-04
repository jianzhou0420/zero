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

from zero.v2.models.lotus.utils.robot_box import RobotBox
from scipy.special import softmax
import torch

from zero.v2.models.lotus.utils.rotation_transform import (
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
        self.config = config.TRAIN_DATASET

    def _remove_table(self, xyz, rgb):
        TABLE_HEIGHT = 0.7505
        table_mask = xyz[:, 2] > TABLE_HEIGHT
        xyz = xyz[table_mask]
        rgb = rgb[table_mask]
        return xyz, rgb

    def _remove_robot(self, xyz, rgb, arm_links_info, rm_robot_type='box_keep_gripper'):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False

        # main
        # get all points belongs to robot
        robot_box = RobotBox(arm_links_info, keep_gripper=keep_gripper, env_name='rlbench', selfgen=self.selfgen)

        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)

        robot_point_ids = np.array(list(robot_point_ids))

        mask = np.ones((xyz.shape[0], ), dtype=bool)

        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False

        # apply mask
        xyz = xyz[mask]
        rgb = rgb[mask]
        # /main

        return xyz, rgb

    def downsample(self, xyz, rgb, action_current, sample_points_by_distance, num_points=4096, downsample_type='random'):
        if len(xyz) > num_points:  # downsample
            if sample_points_by_distance:
                xyz, rgb = self._downsample_by_distance(xyz, rgb, action_current, num_points=num_points)
            else:
                xyz, rgb = self._downsample_random(xyz, rgb, action_current, num_points=num_points)
        else:
            if self.config.same_npoints_per_example:
                point_idxs = np.random.choice(xyz.shape[0], self.num_points, replace=True)
            else:
                max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
                point_idxs = np.random.permutation(len(xyz))[:max_npoints]
            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]

        return xyz, rgb

    def dict_pos_probs(self,):
        pass

    def _remove_outliers(self, xyz, rgb=None, return_idxs=False):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd, idxs = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd, idxs = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
        clf = LocalOutlierFactor(n_neighbors=self.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        if return_idxs:
            return xyz, rgb, idxs
        else:
            return xyz, rgb

    def _downsample_random(self, xyz, rgb, action_current, num_points=4096):
        point_idxs = np.random.choice(len(xyz), num_points, replace=False)
        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        return xyz, rgb

    def _downsample_by_distance(self, xyz, rgb, action_current, num_points=4096):
        dists = np.sqrt(np.sum((xyz - action_current[:3])**2, 1))
        probs = 1 / np.maximum(dists, 0.1)
        probs = np.maximum(softmax(probs), 1e-30)
        probs = probs / sum(probs)
        # probs = 1 / dists
        # probs = probs / np.sum(probs)
        point_idxs = np.random.choice(len(xyz), num_points, replace=False, p=probs)

        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        return xyz, rgb

    def augment_pc(self, flag, xyz, action_current, action_next, gt_rot, aug_max_rot, rot_type, euler_resolution):
        '''
        this is original function from 3dlotus
        it does self.augment_xyz, self.augment_action, self.quaternion_to_type
        '''
        if flag:
            angle = np.random.uniform(-1, 1) * aug_max_rot
            xyz = random_rotate_z(xyz, angle=angle)
            action_current[:3] = random_rotate_z(action_current[:3], angle=angle)
            action_next[:3] = random_rotate_z(action_next[:3], angle=angle)
            action_current[3:-1] = self.rotate_gripper(action_current[3:-1], angle)
            action_next[3:-1] = self.rotate_gripper(action_next[3:-1], angle)

            if rot_type == 'quat':
                gt_rot = action_next[3:-1]
            elif rot_type == 'euler':
                gt_rot = self.rotation_transform.quaternion_to_euler(
                    torch.from_numpy(action_next[3:-1][None, :]))[0].numpy() / 180.
            elif rot_type == 'euler_disc':
                gt_rot = quaternion_to_discrete_euler(action_next[3:-1], euler_resolution)
            elif rot_type == 'rot6d':
                gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                    torch.from_numpy(action_next[3:-1][None, :]))[0].numpy()

            # add small noises (+-2mm)
            pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
            xyz = pc_noises + xyz

            return xyz, action_current, action_next, gt_rot
        else:
            return xyz, action_current, action_next, gt_rot

    def augment_xyz(self, xyz, angle):
        xyz = random_rotate_z(xyz, angle=angle)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz
        return xyz

    def augment_action(self, action, angle):
        action[:3] = random_rotate_z(action[:3], angle)
        action[3:-1] = self.rotate_gripper(action[3:-1], angle)
        return action

    def quaternion_to_type(self, rot, rot_type):
        if rot_type == 'quat':
            gt_rot = rot
        elif rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(rot[None, :]))[0].numpy() / 180.
        elif rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(rot, self.config.euler_resolution)
        elif rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(rot[None, :]))[0].numpy()
        else:
            raise NotImplementedError(f'rot_type {rot_type} not implemented')
        return gt_rot

    def normalize_pc(self, xyz_shift, xyz_norm, xyz, action_current, action_next, gt_rot, height):
        if xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif xyz_shift == 'gripper':
            centroid = copy.deepcopy(action_current[:3])

        if xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        action_next[:3] = (action_next[:3] - centroid) / radius
        action_current[:3] = (action_current[:3] - centroid) / radius
        action_next = np.concatenate([action_next[:3], gt_rot, action_next[-1:]], 0)

        return centroid, radius, height, xyz, action_current, action_next

    def get_robot_point_idxs(self, pos_heatmap_no_robot, xyz, links_info):
        if pos_heatmap_no_robot:
            robot_box = RobotBox(arm_links_info=links_info, env_name='rlbench', selfgen=self.selfgen)
            robot_point_idxs = np.array(
                list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
            )
        else:
            robot_point_idxs = None
        return robot_point_idxs

    def rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot

    def get_groundtruth_rotations(self, action,):
        gt_rots = torch.from_numpy(action.copy())   # quaternions
        rot_type = self.config.rot_type
        euler_resolution = self.config.euler_resolution
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

    def _points_in_workspace(self, xyz, rgb):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])

        xyz = xyz[in_mask]
        rgb = rgb[in_mask]
        return xyz, rgb

    def process_pc(self, xyz, rgb, arm_links_info, voxel_size):
        '''
        params:
            - xyz: (npoints, 3)
            - rgb: (npoints, 3)
            - action_current: (7, )
            - arm_links_info: (bbox, pose)
            - config: config
            - is_train: bool
            - voxel_size: float
        this function do the following:

        1. remove outside workspace
        2. remove robot
        3. remove table
        4. voxelization
        4. remove outliers

        return: pc_ft, action_current

        '''
        rgb = rgb.reshape(-1, 3)
        xyz = xyz.reshape(-1, 3)

        # apply process
        xyz, rgb = self._points_in_workspace(xyz, rgb)
        xyz, rgb = self._remove_robot(xyz, rgb, arm_links_info, rm_robot_type='box_keep_gripper')    # remove robot
        xyz, rgb = self._remove_table(xyz, rgb)    # remove table
        xyz, rgb = self.voxelization(xyz, rgb, voxel_size)
        xyz, rgb = self._remove_outliers(xyz, rgb)  # remove outliers

        return xyz, rgb

    def voxelization(self, xyz, rgb, voxel_size):
       # voxelization as downsample
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.colors = o3d.utility.Vector3dVector(rgb)
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        # # center of the voxel
        # xyz = []
        # rgb = []
        # for voxel in voxel_grid.get_voxels():
        #     xyz.append(voxel.grid_index * voxel_grid.voxel_size + voxel_grid.origin)
        #     rgb.append(voxel.color)
        # xyz = np.array(xyz)
        # rgb = np.array(rgb)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]

        return xyz, rgb

    def action_next_pos_prob(self, xyz, gt_pos, pos_bin_size=0.01, pos_bins=50, heatmap_type='plain', robot_point_idxs=None):
        '''
        heatmap_type:
            - plain: the same prob for all voxels with distance to gt_pos within pos_bin_size
            - dist: prob for each voxel is propotional to its distance to gt_pos
        '''

        shift = np.arange(-pos_bins, pos_bins) * pos_bin_size  # (pos_bins*2, )
        cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None]  # (npoints, 3, pos_bins*2)
        dists = np.abs(gt_pos[None, :, None] - cands_pos)  # (npoints, 3, pos_bins*2)
        dists = einops.rearrange(dists, 'n c b -> c (n b)')  # (3, npoints*pos_bins*2)

        if heatmap_type == 'plain':
            disc_pos_prob = np.zeros((3, xyz.shape[0] * pos_bins * 2), dtype=np.float32)
            disc_pos_prob[dists < 0.01] = 1
            if robot_point_idxs is not None and len(robot_point_idxs) > 0:
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
                disc_pos_prob[:, robot_point_idxs] = 0
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
            for i in range(3):
                if np.sum(disc_pos_prob[i]) == 0:
                    disc_pos_prob[i, np.argmin(dists[i])] = 1
            disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)
            # disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b')
        else:
            disc_pos_prob = 1 / np.maximum(dists, 1e-4)
            # TODO
            # disc_pos_prob[dists > 0.02] = 0
            disc_pos_prob[dists > 0.01] = 0
            if robot_point_idxs is not None and len(robot_point_idxs) > 0:
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
                disc_pos_prob[:, robot_point_idxs] = 0
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
            for i in range(3):
                if np.sum(disc_pos_prob[i]) == 0:
                    disc_pos_prob[i, np.argmin(dists[i])] = 1
            disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)

        return disc_pos_prob

    def obs_to_batch(self, obs):
        '''

        '''

        pass

    def pc_action_standard_process(self, xyz, rgb, arm_links_info, action_current, action_next=None, is_train=False):
        '''
        overall process:
        1. remove outside workspace
        2. remove robot
        3. remove table
        4. remove outliers
        5. voxelization
        6. downsample
        7. augment
        8. normalize
        '''

        voxel_size = 0.005
        # 1.remove outside workspace 2.remove robot 3.remove table 4.remove outliers 5.voxelization
        xyz, rgb = self.process_pc(xyz, rgb, arm_links_info, voxel_size)
        xyz, rgb = self.downsample(xyz, rgb, action_current, sample_points_by_distance=self.config.sample_points_by_distance, num_points=self.config.num_points, downsample_type='random')

        robot_box = RobotBox(arm_links_info=arm_links_info, env_name='rlbench', selfgen=self.selfgen)
        robot_point_idxs = np.array(list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1]))  # 需要放在pc缩减之后，augment之前
        height = xyz[:, -1] - 0.7505  # 相当于每个点对于桌面的高度，其实我觉得应该放在augment之后，不过先按照作者的思路来。

        # if is_train:  # 7.augment
        #     angle = np.random.uniform(-1, 1) * self.config.aug_max_rot
        #     xyz = self.augment_xyz(xyz, angle)
        #     action_current = self.augment_action(action_current, angle)
        #     action_next = self.augment_action(action_next, angle)
        #     pass

        # 8.normalize
        centroid = np.mean(xyz, axis=0)
        radius = 1
        xyz = (xyz - centroid) / radius
        action_current[:3] = (action_current[:3] - centroid) / radius
        rgb = rgb / 255.0 * 2 - 1

        if is_train:
            action_next[:3] = (action_next[:3] - centroid) / radius

        # post-process
        # 1.convert action_next's quaternion to discrete euler
            action_next_rot = quaternion_to_discrete_euler(action_next[3:-1], self.config.euler_resolution)
            action_next = np.concatenate([action_next[:3], action_next_rot, action_next[-1:]], axis=0)  # attention rot_type changed
        # 2.get gt_pos_prob 是最后再做的，约等于转换以下action_next
            disc_pos_prob = self.action_next_pos_prob(
                xyz, action_next[:3], pos_bins=self.config.pos_bins,
                pos_bin_size=self.config.pos_bin_size,
                heatmap_type=self.config.pos_heatmap_type,
                robot_point_idxs=robot_point_idxs
            )
        else:
            disc_pos_prob = None

        pc_ft = np.concatenate((xyz, rgb, height[:, None]), axis=1)

        return pc_ft, action_current, action_next, centroid, radius, disc_pos_prob

    def dataset_generation_single_episodes(self, data, voxel_size, episode_path, instr_embeds, taskvar_instrs, visualize=False):
        ''' #TODO: voxel_size is not used
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
            'data_ids': [],
            'pc_fts': [],
            'step_ids': [],
            'pc_centroids': [],
            'pc_radius': [],
            'ee_poses': [],
            'txt_embeds': [],
            'gt_actions': [],
            'disc_pos_probs': []
        }

        all_names = episode_path.split('/')
        task_name = all_names[9]
        variation_name = all_names[10].split('variation')[-1]
        episode_name = all_names[12]
        taskvar = f'{task_name}_peract+{variation_name}'

        for t in range(len(data['key_frameids']) - 1):  # last frame dont train
            xyz = data['pc'][t]
            rgb = data['rgb'][t]
            arm_links_info = (data['bbox'][t], data['pose'][t])
            action_current = data['action'][t]
            action_next = data['action'][t + 1]

            # language embedding
            instr_embed = instr_embeds[random.choice(taskvar_instrs[taskvar])]

            # pointcloud process
            pc_ft, action_current, action_next, centroid, radius, disc_pos_prob = self.pc_action_standard_process(
                xyz, rgb, arm_links_info, action_current, action_next, is_train=True
            )
            outs['disc_pos_probs'].append(disc_pos_prob)
            outs['data_ids'].append(f'{task_name}-{variation_name}-{episode_name}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(action_current).float())
            outs['gt_actions'].append(torch.from_numpy(action_next).float())
            outs['step_ids'].append(t)
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)
        return outs

    def retrieve_all_episodes(self, root_path):
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

    def dataset_generation(self, origin_data_root, output_dir, tasks_to_use=None):
        voxel_size = 0.005
        tasks_list = self.retrieve_all_episodes(origin_data_root)
        taskvar_instr_file = self.config.taskvar_instr_file
        instr_embed_file = self.config.instr_embed_file
        taskvar_instrs = json.load(open(taskvar_instr_file))
        instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        if self.config.instr_embed_type == 'last':
            instr_embeds = {instr: embeds[-1:] for instr, embeds in instr_embeds.items()}
        total = sum([len(variation) for task in tasks_list for variation in task])    # nested sum
        pbar = tqdm(total=total)
        for task in tasks_list:
            task_name = task[0][0].split('/')[9]
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
                    new_data = self.dataset_generation_single_episodes(data, voxel_size, episode, instr_embeds, taskvar_instrs, visualize=False)

                    with open(export_path, 'wb') as f:
                        pickle.dump(new_data, f)
                    pbar.update(1)


if __name__ == '__main__':
    config_path = '/data/zero/zero/v2/config/lotus_exp2_0.005_close_jar.yaml'
    config = yacs.config.CfgNode(new_allowed=True)
    config.merge_from_file(config_path)
    op = ObsProcessLotus(config, selfgen=True)

    origin_data_root = '/data/selfgen/20250105/train_dataset/keysteps/seed42'
    # from datetime import datetime

    output_dir = f'/data/selfgen/voxellizationexp_outlier/'
    tasks_to_use = ['close_jar']
    op.dataset_generation(origin_data_root, output_dir, tasks_to_use=tasks_to_use)
