

from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
import re
import open3d as o3d
import numpy as np
import torch
from copy import deepcopy as copy
import einops
import pickle
import json
from numpy import array as npa
from pathlib import Path
import random
from typing import Dict, Optional, Sequence
from collections import defaultdict, Counter
from zero.expForwardKinematics.models.lotus.utils.robot_box import RobotBox
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import *
import torchvision.transforms.functional as transforms_f
from zero.expForwardKinematics.models.lotus.utils.rotation_transform import quaternion_to_discrete_euler, RotationMatrixTransform
from zero.expForwardKinematics.ReconLoss.FrankaPandaFK import FrankaEmikaPanda
import torchvision.transforms as transforms
from codebase.z_utils.open3d import *
from codebase.z_utils.idx_mask import *
from codebase.z_utils.Rotation import quat2euler, euler2quat
from scipy.spatial.transform import Rotation as R
from typing_extensions import override


class ObsProcessorDP3(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.train_flag = train_flag
        self.WORKSPACE = get_robot_workspace(real_robot=False, use_vlm=False)
        self.dataset_init_flag = False
        self.train_flag = train_flag
        self.franka = FrankaEmikaPanda()

    @override
    def static_process(self, obs_raw):
        out = {
            'xyz': [],
            'rgb': [],
            'eePose_hist': [],
            'eePose_futr': [],
            'JP_hist': [],
            'JP_futr': []
        }

        # action
        if self.train_flag:
            pass
            action_all = obs_raw['eePose_all']
            JP_all = obs_raw['JP_all']

            # save path
            num_frames = len(obs_raw['rgb']) - 1

            # actions
            for i in range(num_frames):
                keyframe_id = copy(np.array(obs_raw['key_frameids'][i], dtype=np.int16))

                action_curr = copy(np.array(obs_raw['eePose'][i], dtype=np.float64))
                action_next = copy(np.array(obs_raw['eePose'][i + 1], dtype=np.float64))
                action_path = copy(np.array(action_all[obs_raw['key_frameids'][i]:obs_raw['key_frameids'][i + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

                open_all = np.array([a[7] for a in action_all])
                JP_all_copy = copy(JP_all)
                JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

                JP_curr = copy(np.array(JP_all_copy[obs_raw['key_frameids'][i]], dtype=np.float64))
                JP_next = copy(np.array(JP_all_copy[obs_raw['key_frameids'][i + 1]], dtype=np.float64))
                JP_path = copy(np.array(JP_all_copy[obs_raw['key_frameids'][i]:obs_raw['key_frameids'][i + 1] + 1], dtype=np.float64))

                # eePose_hist
                if keyframe_id - 8 < 0:
                    eePose_hist = [action_all[j] for j in range(keyframe_id)]
                    eePose_hist += [action_curr] * (8 - keyframe_id)

                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id)]
                    JP_hist += [JP_curr] * (8 - keyframe_id)
                else:
                    eePose_hist = [action_all[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                # eePose_futr
                eePose_futr, JP_futr = self.find_middle_actions(action_path, JP_path)

                # concatenate
                eePose_hist = np.stack(eePose_hist, axis=0)
                eePose_futr = np.stack(eePose_futr, axis=0)
                JP_hist = np.stack(JP_hist, axis=0)
                JP_futr = np.stack(JP_futr, axis=0)
                # check & save
                assert np.allclose(action_curr, eePose_hist[-1])
                assert np.allclose(action_next, eePose_futr[-1])
                assert np.allclose(action_curr, eePose_hist[-1])

                assert np.allclose(JP_curr, JP_all_copy[keyframe_id])
                assert np.allclose(JP_next, JP_futr[-1])
                assert np.allclose(JP_curr, JP_hist[-1])

                out['eePose_hist'].append(eePose_hist)
                out['eePose_futr'].append(eePose_futr)
                out['JP_hist'].append(JP_hist)
                out['JP_futr'].append(JP_futr)
        else:

            eePose_hist = copy(obs_raw['eePose_hist_eval'])
            JP_hist = copy(obs_raw['JP_hist_eval'])
            out['JP_hist'].append(JP_hist)
            out['eePose_hist'].append(eePose_hist)

        if self.train_flag:
            num_keyframes_with_end = len(obs_raw['key_frameids'])
            num_keyframes = num_keyframes_with_end - 1
        else:
            num_keyframes_with_end = 1
            num_keyframes = 1

        VoxelGrid_list = []

        for t in range(num_keyframes_with_end):  # voxelize first
            arm_links_info = (obs_raw['bbox'][t], obs_raw['pose'][t])
            xyz = obs_raw['xyz'][t].reshape(-1, 3)
            rgb = obs_raw['rgb'][t].reshape(-1, 3)

            # 1. within workspace
            in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) &\
                (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) &\
                (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
            # 2. remove table
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
            xyz = xyz[in_mask]
            rgb = rgb[in_mask]

            # 3. voxelize
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd, _, trace = pcd.voxel_down_sample_and_trace(
            #     self.config.Dataset.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
            # )
            # xyz = np.asarray(pcd.points)
            # trace = np.array([v[0] for v in trace])
            # rgb = rgb[trace]

            # 3. remove robot get gripper idx
            JP_curr = copy(np.array(obs_raw['JP_curr_no_open'][t], dtype=np.float64))
            JP_curr = np.concatenate([JP_curr, np.array([obs_raw['eePose'][t][-1]])], axis=0)
            mask = self._rm_robot_by_JP(xyz, JP_curr)
            xyz = xyz[~mask]
            rgb = rgb[~mask]

            # 4. FK voxelization
            VoxelGrid = o3d.geometry.VoxelGrid()
            VoxelGrid.voxel_size = self.config.Dataset.voxel_size
            VoxelGrid.origin = np.array([0, 0, 0])
            # xyz(555,3)
            voxel_index_set = set()
            voxel_points_dict = {}

            # 算index
            for point_idx, xyz_s in enumerate(xyz):
                grid_index = tuple(np.floor((xyz_s - VoxelGrid.origin) / VoxelGrid.voxel_size).astype(int))
                voxel_index_set.add(grid_index)

                if grid_index not in voxel_points_dict:
                    voxel_points_dict[grid_index] = []
                voxel_points_dict[grid_index].append(point_idx)

            # 创建voxel,这里先不考虑augmentation的问题
            for voxel_idx in voxel_index_set:
                voxel_xyz = xyz[voxel_points_dict[voxel_idx]]
                voxel_rgb = rgb[voxel_points_dict[voxel_idx]]
                voxel_s = o3d.geometry.Voxel(grid_index=voxel_idx, color=voxel_rgb[0],)
                VoxelGrid.add_voxel(voxel_s)
            VoxelGrid_list.append(VoxelGrid)

        new_out = {
            'obs': {
                'point_cloud': out['xyz'],
                'agent_pose': out['eePose_hist'],
            },
            'action': out['eePose_futr'],
        }
        return new_out

    @override
    def dynamic_process(self, data, *args, **kwargs):
        n_frames = len(data['rgb'])

        outs = {
            'obs': {
                'point_cloud': [],
                'agent_pose': [],
            },
            'action': [],
        }
        # dynamic process
        for i in range(n_frames):
            # 1. retrieve data
            xyz = npa(copy(data['xyz'][i]))
            rgb = npa(copy(data['rgb'][i]))
            JP_hist = npa(copy(data['JP_hist'][i]))
            height = np.expand_dims(npafp32(copy(xyz[:, 2])), axis=1)
            height = (height - self.TABLE_HEIGHT)

            if self.config['TrainDataset']['augmentation'] is True:
                # 2. downsample by number
                idx = pcd_random_downsample_by_num(xyz, rgb, num_points=self.num_points, return_idx=True)
                xyz = xyz[idx]
                rgb = rgb[idx]
                height = height[idx]
                if self.train_flag is True:
                    noncollision_mask = noncollision_mask[idx]
            else:
                pass
                # 3. normalize point cloud
            center = np.mean(xyz, 0)
            xyz = xyz - center
            rgb = (rgb / 255.0) * 2 - 1
            pc_fts = np.hstack([xyz, rgb, height])  # (N, 6)

            # normalize joint positions
            JP_hist = normalize_JP(JP_hist)
            if self.train_flag is True:
                JP_futr = tensorfp32(copy(data['JP_futr'][i]))
                JP_futr = normalize_JP(JP_futr)
                outs['JP_futr'].append(JP_futr)

            outs['pc_fts'].append(tensorfp32(pc_fts))
            outs['JP_hist'].append(tensorfp32(JP_hist))

        return outs

    @override
    def dataset_init(self):
        config = self.config
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self._resize = Resize(config['TrainDataset']['image_rescales'])

    @override
    @staticmethod
    def collate_fn(batch):
        collated = {
            'obs': {},
            'action': None,
        }
        for key in batch[0]['obs'].keys():
            # Concatenate the tensors from each dict in the batch along dim=0.
            collated['obs'][key] = torch.cat([minibatch['obs'][key] for minibatch in batch], dim=0)
        try:
            collated['action'] = torch.cat([minibatch['action'] for minibatch in batch], dim=0)
        except:
            pass
        return collated

    # utils
    def find_middle_actions(self, actions_path, theta_actions_path, horizon=8):
        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    @override
    def denormalize_action(self, action: dict) -> list:
        '''
        assume action has shape (1, H, D), batch size, horizon, action dim
        '''
        action = action['action_pred']
        B, H, D = action.shape
        new_action = np.zeros((B, H, 8), dtype=np.float32)
        new_action[:, :, :3] = denormalize_pos(action[:, :, :3]).cpu().detach().numpy()
        angles = action[:, :, 3:6].cpu().detach().numpy() * 3.15

        angles = einops.rearrange(angles, 'b h d -> (b h) d')
        angles = euler2quat(angles)
        angles = einops.rearrange(angles, '(b h) d -> b h d', b=B, h=H)
        new_action[:, :, 3:7] = angles
        new_action[:, :, 7:8] = action[:, :, 6:7].cpu().detach().numpy()

        new_action = [new_action[0, i, :] for i in range(H)]

        return new_action

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

    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot

    def _find_gt_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg'):
        if sub_keyframe_dection_mode == 'avg':
            indices = np.linspace(0, len(actions_path) - 1, self.config.horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
            gt_actions = [actions_path[i] for i in indices]
            gt_theta_actions = [theta_actions_path[i] for i in indices]
            return gt_actions, gt_theta_actions
        elif sub_keyframe_dection_mode == 'xyzpeak':
            NotImplementedError("XYZPEAK")

    def _dataset_init_FK(self):
        config = self.config
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self.num_points = config['TrainDataset']['num_points']
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=False)['TABLE_HEIGHT']

    def _rm_robot_by_JP(self, xyz, JP):
        theta = JP - self.franka.JP_offset
        bbox_link, bbox_other = self.franka.theta2obbox(theta)
        bbox_all = bbox_link + bbox_other[:2]
        pcd_idx = get_robot_pcd_idx(xyz, bbox_all)
        return pcd_idx

    def get_uncollision_mask(self, xyz, JP):
        theta = JP - self.franka.JP_offset
        bbox_link, bbox_other = self.franka.theta2obbox(theta)
        gripper_idx = get_robot_pcd_idx(xyz, *[bbox_other[2:]])
        return gripper_idx

    def find_middle_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg', horizon=8):
        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    def within_workspace(self, xyz, rgb):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) &\
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) &\
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        xyz = xyz[in_mask]
        rgb = rgb[in_mask]
        return xyz, rgb

    def remove_table(self, xyz, rgb):
        in_mask = xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT']
        xyz = xyz[in_mask]
        rgb = rgb[in_mask]
        return xyz, rgb

    def voxelize(self, xyz, rgb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.Dataset.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        return xyz, rgb


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
