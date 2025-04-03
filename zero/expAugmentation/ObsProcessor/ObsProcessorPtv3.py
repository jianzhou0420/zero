import re
from zero.expAugmentation.ObsProcessor.ObsProcessorBase import ObsProcessorBase
import copy
import open3d as o3d
import numpy as np
import torch
from zero.expAugmentation.models.lotus.utils.rotation_transform import quaternion_to_discrete_euler, RotationMatrixTransform
import einops
# from rlbench.demo import Demo
from tqdm import tqdm
import os
import pickle
import json
from PIL import Image

import numpy as np
import random
import collections

from zero.expAugmentation.models.lotus.utils.robot_box import RobotBox
import numpy as np

from zero.z_utils.joint_position import normaliza_JP
import json
from scipy.spatial.transform import Rotation as R

from zero.dataprocess.utils import convert_gripper_pose_world_to_image, keypoint_discovery
from zero.expAugmentation.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
from zero.expAugmentation.ReconLoss.ForwardKinematics import FrankaEmikaPanda

# --------------------------------------------------------------
# region utils

JOINT_POSITIONS_LIMITS = np.array([[-2.8973, 2.8973],
                                   [-1.7628, 1.7628],
                                   [-2.8973, 2.8973],
                                   [-3.0718, -0.0698],
                                   [-2.8973, 2.8973],
                                   [-0.0175, 3.7525],
                                   [-2.8973, 2.8973]])


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


# endregion
# --------------------------------------------------------------
# region main logic


class ObsProcessorPtv3(ObsProcessorBase):
    def __init__(self, config, train_flag=True):
        '''
        simulator generate demo or obs
        1. demo or obs 2 obs_raw
        2. obs_static_process
        3. obs_dynamic_process
        4. collate_fn 2 batch
        '''
        super().__init__(config)
        self.config = config
        self.rotation_transform = RotationMatrixTransform()
        self.WORKSPACE = get_robot_workspace(real_robot=False, use_vlm=False)
        self.dataset_init_flag = False
        self.train_flag = train_flag
        self.franka = FrankaEmikaPanda()

        # region obs_raw

    def obs_2_obs_raw(self, obs):
        key_frames = [0]
        state_dict = self.obs2dict(obs)

        bbox = []
        pose = []

        single_bbox = dict()
        single_pose = dict()
        for key, value in obs.misc.items():
            if key.split('_')[-1] == 'bbox':
                single_bbox[key.split('_bbox')[0]] = value
            if key.split('_')[-1] == 'pose':
                single_pose[key.split('_pose')[0]] = value
        bbox.append(single_bbox)
        pose.append(single_pose)

        actions = []
        positions = []
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)
        position = obs.joint_positions
        actions.append(action)
        positions.append(position)
        obs_raw = {
            'key_frameids': key_frames,
            'rgb': state_dict['rgb'],  # (T, N, H, W, 3)
            'pc': state_dict['pc'],  # (T, N, H, W, 3)
            'action': state_dict['gripper'],  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
        }
        return obs_raw

    def demo_2_obs_raw(self, demo):
        """Fetch the desired state based on the provided demo.
        :param obs: incoming obs
        :return: required observation (rgb, depth, pc, gripper state)
        """

        key_frames = keypoint_discovery(demo)
        key_frames.insert(0, 0)

        state_dict_ls = collections.defaultdict(list)
        for f in key_frames:
            state_dict = self.obs2dict(demo._observations[f])
            for k, v in state_dict.items():
                if len(v) > 0:
                    # rgb: (N: num_of_cameras, H, W, C); gripper: (7+1, )
                    state_dict_ls[k].append(v)

        for k, v in state_dict_ls.items():
            state_dict_ls[k] = np.stack(v, 0)  # (T, N, H, W, C)

        action_ls = state_dict_ls['gripper']  # (T, 7+1)
        del state_dict_ls['gripper']

        # return demo, key_frames, state_dict_ls, action_ls

        gripper_pose = []
        for key_frameid in key_frames:
            gripper_pose.append(demo[key_frameid].gripper_pose)

        # get bbox and poses of each link
        bbox = []
        pose = []
        for key_frameid in key_frames:
            single_bbox = dict()
            single_pose = dict()
            for key, value in demo[key_frameid].misc.items():
                if key.split('_')[-1] == 'bbox':
                    single_bbox[key.split('_bbox')[0]] = value
                if key.split('_')[-1] == 'pose':
                    single_pose[key.split('_pose')[0]] = value
            bbox.append(single_bbox)
            pose.append(single_pose)

        # get actions_all
        # get positions_all
        actions = []
        positions = []
        for obs in demo._observations:
            action = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)
            position = obs.joint_positions
            actions.append(action)
            positions.append(position)

        obs_raw = {
            'key_frameids': key_frames,
            'rgb': state_dict_ls['rgb'],  # (T, N, H, W, 3)
            'pc': state_dict_ls['pc'],  # (T, N, H, W, 3)
            'action': action_ls,  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
            'sem': state_dict_ls['sem'],  # (T, N, H, W, 3)
            'actions_all': actions,
            'joint_position_all': positions,
        }
        return obs_raw
        # endregion
    # region static process

    def static_process_fk(self, obs_raw, taskvar):
        '''
        承接原始数据，处理成为简单的格式，给dataset处理。
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

        obs_static_process = {
            'xyz': [],
            'rgb': [],
            'eePose_hist': [],
            'eePose_futr': [],
            'JP_hist': [],
            'JP_futr': [],
        }

        # all_names = episode_path.split('/')
        # task_name = all_names[6]
        # variation_name = all_names[7].split('variation')[-1]
        # episode_name = all_names[9]
        # taskvar = f'{task_name}_peract+{variation_name}'

        '''
        1. remove outside workspace
        2. remove table
        3. voxelization
        '''

        if self.train_flag:
            num_keyframes = len(obs_raw['key_frameids']) - 1
        else:
            num_keyframes = 1

        for t in range(num_keyframes):  # obs itself
            arm_links_info = (obs_raw['bbox'][t], obs_raw['pose'][t])
            xyz = obs_raw['pc'][t].reshape(-1, 3)
            rgb = obs_raw['rgb'][t].reshape(-1, 3)

            # 1. within workspace
            in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                      (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                      (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
            # 2. remove table
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
            xyz = xyz[in_mask]
            rgb = rgb[in_mask]

            JP_curr = copy.deepcopy(np.array(obs_raw['joint_position_all'][obs_raw['key_frameids'][t]], dtype=np.float64))
            JP_curr = np.concatenate([JP_curr, [obs_raw['action'][t][-1]]], axis=0)
            mask = self._rm_robot_by_JP(xyz, JP_curr)

            xyz = xyz[~mask]
            rgb = rgb[~mask]

            # 3. voxelize
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd, _, trace = pcd.voxel_down_sample_and_trace(
                self.config.Dataset.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
            )
            xyz = np.asarray(pcd.points)
            trace = np.array([v[0] for v in trace])
            rgb = rgb[trace]

            # 4. remove outliers
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)

            xyz = xyz[ind]
            rgb = rgb[ind]

            obs_static_process['xyz'].append(xyz)
            obs_static_process['rgb'].append(rgb)

        # only for train
        if not self.train_flag:  # flow control
            return obs_static_process

        action_all = obs_raw['actions_all']
        JP_all = obs_raw['joint_position_all']
        for t in range(num_keyframes):
            keyframe_id = copy.deepcopy(np.array(obs_raw['key_frameids'][t], dtype=np.int16))
            rgb = obs_raw['rgb'][t]
            pcd = obs_raw['pc'][t]

            eePose_curr = copy.deepcopy(np.array(obs_raw['action'][t], dtype=np.float64))
            eePose_next = copy.deepcopy(np.array(obs_raw['action'][t + 1], dtype=np.float64))
            eePose_path = copy.deepcopy(np.array(action_all[obs_raw['key_frameids'][t]:obs_raw['key_frameids'][t + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

            open_all = np.array([a[7] for a in action_all])
            JP_all_copy = copy.deepcopy(JP_all)
            JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

            JP_curr = copy.deepcopy(np.array(JP_all_copy[obs_raw['key_frameids'][t]], dtype=np.float64))
            JP_next = copy.deepcopy(np.array(JP_all_copy[obs_raw['key_frameids'][t + 1]], dtype=np.float64))
            JP_path = copy.deepcopy(np.array(JP_all_copy[obs_raw['key_frameids'][t]:obs_raw['key_frameids'][t + 1] + 1], dtype=np.float64))

            # action_history
            if keyframe_id - 8 <= 1:
                eePose_hist = [action_all[j] for j in range(keyframe_id)]
                eePose_hist += [eePose_curr] * (8 - keyframe_id)

                JP_hist = [JP_all_copy[j] for j in range(keyframe_id)]
                JP_hist += [JP_curr] * (8 - keyframe_id)
            else:
                eePose_hist = [action_all[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                JP_hist = [JP_all_copy[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
            # action_future
            eePose_futr, JP_futr = self.find_middle_actions(eePose_path, JP_path, sub_keyframe_dection_mode='avg')

            # concatenate
            eePose_hist = np.stack(eePose_hist, axis=0)
            eePose_futr = np.stack(eePose_futr, axis=0)

            JP_hist = np.stack(JP_hist, axis=0)
            JP_futr = np.stack(JP_futr, axis=0)
            # check & save
            assert np.allclose(eePose_curr, eePose_hist[-1])
            assert np.allclose(eePose_next, eePose_futr[-1])
            assert np.allclose(eePose_curr, eePose_hist[-1])

            assert np.allclose(JP_curr, JP_all_copy[keyframe_id])
            assert np.allclose(JP_next, JP_futr[-1])
            assert np.allclose(JP_curr, JP_hist[-1])

            obs_static_process['eePose_hist'].append(eePose_hist)
            obs_static_process['eePose_futr'].append(eePose_futr)

            obs_static_process['JP_hist'].append(JP_hist)
            obs_static_process['JP_futr'].append(JP_futr)

        return obs_static_process

    def static_process_FK(self, obs_raw):
        pass
    # endregion
    # region dynamic process

    def dynamic_process(self, obs_static, taskvar):  # TODO: refine
        '''
        obs_static_process = {
            'xyz': [],
            'rgb': [],
            'action_current': [],
            'action_next': [],
            'data_ids': [],
            'arm_links_info': [],
            'actions_path': [],
            'theta_actions_path': [],
        }

        '''
        if self.dataset_init_flag is False:
            self._dataset_init()

        obs_dynamic_out = {
            'data_ids': [],
            'pc_fts': [],
            'step_ids': [],
            'pc_centroids': [],
            'pc_radius': [],
            'ee_poses': [],
            'txt_embeds': [],
            'gt_actions': [],
            'disc_pos_probs': [],
            'theta_positions': [],
        }
        num_frames = len(obs_static['xyz'])
        # 1.get specific frame data

        aug_angle = []  # because I seperate the xyz process and action process, so I need to store the aug_angle for second loop

        for t in range(num_frames):
            sub_keyframe_dection_mode = 'avg'
            assert sub_keyframe_dection_mode in ['avg', 'xyzpeak']

            # end of path processs
            # data_ids = obs_raw['data_ids'][t]
            xyz = copy.deepcopy(obs_static['xyz'][t])
            rgb = copy.deepcopy(obs_static['rgb'][t])

            xyz, rgb = obs_static['xyz'][t], obs_static['rgb'][t]

            # randomly select one instruction
            instr = random.choice(self.taskvar_instrs[taskvar])
            instr_embed = copy.deepcopy(self.instr_embeds[instr])

            # 5. downsample point cloud
            # sampling points

            if len(xyz) > self.num_points:  # 如果不要它，直接num_points=10000000
                tmp_flag = True  # TODO： remove tmp_flag
                point_idxs = np.random.choice(len(xyz), self.num_points, replace=False)
            else:
                tmp_flag = False
                max_npoints = int(len(xyz) * np.random.uniform(0.4, 0.6))
                point_idxs = np.random.permutation(len(xyz))[:max_npoints]

            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]

            height = xyz[:, -1] - self.TABLE_HEIGHT
            # print(f"After downsample xyz: {xyz.shape}")

            # 6. point cloud augmentation

            if self.augment_pc:
                # rotate around z-axis
                aug_angle.append(np.random.uniform(-1, 1) * self.aug_max_rot)
                xyz = random_rotate_z(xyz, angle=aug_angle[t])

            if tmp_flag:
                pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
                xyz = pc_noises + xyz

            # 7. normalize point cloud
            if self.xyz_shift == 'none':
                centroid = np.zeros((3, ))
            elif self.xyz_shift == 'center':
                centroid = np.mean(xyz, 0)
            elif self.xyz_shift == 'gripper':
                centroid = copy.deepcopy(ee_pose_current[:3])
            if self.xyz_norm:
                radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
            else:
                radius = 1

            xyz = (xyz - centroid) / radius
            height = height / radius

            rgb = (rgb / 255.) * 2 - 1
            pc_ft = np.concatenate([xyz, rgb], 1)
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

            # print(f"{taskvar}: {xyz.shape}")
            # outs['data_ids'].append(data_ids)
            obs_dynamic_out['pc_centroids'].append(centroid)
            obs_dynamic_out['pc_radius'].append(radius)
            obs_dynamic_out['pc_fts'].append(torch.from_numpy(pc_ft).float())
            obs_dynamic_out['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            obs_dynamic_out['step_ids'].append(t)

        if self.train_flag is False:
            return obs_dynamic_out

        for t in range(num_frames):
            ee_pose_current = copy.deepcopy(obs_static['action_current'][t])
            ee_pose_next = copy.deepcopy(obs_static['action_next'][t])

            action_path = copy.deepcopy(obs_static['actions_path'][t])
            theta_actions_path = copy.deepcopy(obs_static['theta_actions_path'][t])

            gt_ee_poses, gt_theta_position = self._find_gt_actions(action_path, theta_actions_path, sub_keyframe_dection_mode)
            # assert (gt_actions[0] == ee_pose).all()
            assert (gt_ee_poses[-1] == ee_pose_next).all()
            assert len(gt_ee_poses) == self.config.horizon
            # append open to theta_actions
            for i in range(len(gt_theta_position)):
                gt_theta_position[i] = np.append(gt_theta_position[i], gt_ee_poses[i][-1])

            if self.augment_pc:
                ee_pose_current[:3] = random_rotate_z(ee_pose_current[:3], angle=aug_angle[t])
                ee_pose_current[3:-1] = self._rotate_gripper(ee_pose_current[3:-1], aug_angle[t])
                new_gt_actions = []
                for i, action in enumerate(gt_ee_poses):
                    action[:3] = random_rotate_z(action[:3], angle=aug_angle[t])
                    action[3:-1] = self._rotate_gripper(action[3:-1], aug_angle[t])
                    new_gt_actions.append(action)
                gt_ee_poses = np.stack(new_gt_actions, 0)

                gt_rot = []
                for action in gt_ee_poses:
                    gt_rot.append(quaternion_to_discrete_euler(action[3:-1], self.euler_resolution))
                gt_rot = np.stack(gt_rot, 0)

            centroid_t = obs_dynamic_out['pc_centroids'][t]
            radius_t = obs_dynamic_out['pc_radius'][t]

            # ee_pose_actions
            gt_ee_poses[:, :3] = (gt_ee_poses[:, :3] - centroid_t) / radius_t
            ee_pose_current[:3] = (ee_pose_current[:3] - centroid_t) / radius_t
            gt_ee_poses = np.concatenate([gt_ee_poses[:, :3], gt_rot, gt_ee_poses[:, -1:]], 1)

            # theta_actions
            test = torch.from_numpy(np.array(gt_theta_position)).float()
            test = normaliza_JP(test)
            test = einops.rearrange(test, 'h a -> a h')  # 现在channel是各个纬度的action

            obs_dynamic_out['ee_poses'].append(torch.from_numpy(ee_pose_current).float())
            obs_dynamic_out['gt_actions'].append(torch.from_numpy(gt_ee_poses).float())
            obs_dynamic_out['theta_positions'].append(torch.from_numpy(np.array(test)).float())

        obs_dynamic_out = obs_dynamic_out

        return obs_dynamic_out
    # endregion
    # region collate_fn

    def get_collect_function(self):
        def ptv3_collate_fn(data):
            batch = {}
            if type(data) == dict:
                data = [data]

            for key in obs_raw[0].keys():
                batch[key] = sum([x[key] for x in data], [])

            npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
            batch['npoints_in_batch'] = npoints_in_batch
            batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
            batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)  # (#all points, 6)

            for key in ['ee_poses', 'gt_actions', 'theta_positions']:
                if len(batch[key]) != 0:
                    batch[key] = torch.stack(batch[key], 0)

            # if 'disc_pos_probs' in batch:
            #     batch['disc_pos_probs'] = batch['disc_pos_probs'] # [(3, #all pointspos_bins*2)]

            batch['step_ids'] = torch.LongTensor(batch['step_ids'])

            batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
            batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

            if len(batch['pc_centroids']) > 0:
                batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

            return batch
        return ptv3_collate_fn
    # endregion
    # private functions

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

    def obs2dict(self, obs):
        apply_rgb = True
        apply_pc = True
        apply_cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
        apply_depth = True
        apply_sem = False
        gripper_pose = False
        # fetch state: (#cameras, H, W, C)
        state_dict = {"rgb": [], "depth": [], "pc": [], "sem": []}
        for cam in apply_cameras:
            if apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

            if apply_sem:
                sem = getattr(obs, "{}_mask".format(cam))
                state_dict["sem"] += [sem]

        # fetch gripper state (3+4+1, )
        gripper = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(
            np.float32
        )
        state_dict["gripper"] = gripper

        if gripper_pose:
            gripper_imgs = np.zeros(
                (len(apply_cameras), 1, 128, 128), dtype=np.float32
            )
            for i, cam in enumerate(apply_cameras):
                u, v = convert_gripper_pose_world_to_image(obs, cam)
                if u > 0 and u < 128 and v > 0 and v < 128:
                    gripper_imgs[i, 0, v, u] = 1
            state_dict["gripper_imgs"] = gripper_imgs

        state_dict['rgb'] = np.stack(state_dict['rgb'], 0)
        state_dict['depth'] = np.stack(state_dict['depth'], 0)
        state_dict['pc'] = np.stack(state_dict['pc'], 0)

        return state_dict

    def _dataset_init(self):
        self.taskvar_instrs = json.load(open(self.config.TRAIN_DATASET.taskvar_instr_file))
        self.instr_embeds = np.load(self.config.TRAIN_DATASET.instr_embed_file, allow_pickle=True).item()
        # 0.1 Downsample args
        self.num_points = self.config.TRAIN_DATASET.num_points

        # 0.2 shift and normalization
        self.xyz_shift = self.config.TRAIN_DATASET.xyz_shift
        self.xyz_norm = self.config.TRAIN_DATASET.xyz_norm

        # put together
        self.use_height = self.config.TRAIN_DATASET.use_height

        # augment & action head
        self.rot_type = self.config.TRAIN_DATASET.rot_type
        self.augment_pc = self.config.TRAIN_DATASET.augment_pc
        self.aug_max_rot = np.deg2rad(self.config.TRAIN_DATASET.aug_max_rot)
        self.euler_resolution = self.config.TRAIN_DATASET.euler_resolution

        self.pos_type = self.config.TRAIN_DATASET.pos_type
        self.pos_bins = self.config.TRAIN_DATASET.pos_bins
        self.pos_bin_size = self.config.TRAIN_DATASET.pos_bin_size
        self.pos_heatmap_type = self.config.TRAIN_DATASET.pos_heatmap_type
        self.pos_heatmap_no_robot = self.config.TRAIN_DATASET.pos_heatmap_no_robot

        # 0.1. Load some pheripheral information
        self.real_robot = self.config.TRAIN_DATASET.real_robot
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=self.real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()
        self.dataset_init_flag = True

    def _rm_robot_by_JP(self, xyz, JP):
        def get_robot_pcd_idx(xyz, obbox):
            points = o3d.utility.Vector3dVector(xyz)
            robot_point_idx = set()
            for box in obbox:
                tmp = box.get_point_indices_within_bounding_box(points)
                robot_point_idx = robot_point_idx.union(set(tmp))
            robot_point_idx = np.array(list(robot_point_idx))
            mask = np.zeros(len(xyz), dtype=bool)
            mask[robot_point_idx] = True
            return mask

        theta = JP - self.franka.JP_offset
        bbox_link, bbox_other = self.franka.theta2obbox(theta)
        bbox_all = bbox_link + bbox_other[:2]
        pcd_idx = get_robot_pcd_idx(xyz, bbox_all)
        return pcd_idx

    def find_middle_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg', horizon=8):

        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    # region modulization

    def within_workspace(self, xyz, rgb):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                  (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
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
    # endregion

# endregion
# --------------------------------------------------------------
# region test


def test_collect_obs(record_example=False):
    from zero.expAugmentation.config.default import get_config
    import matplotlib.pyplot as plt
    import pickle
    config = get_config('/media/jian/ssd4t/zero/zero/expAugmentation/config/expBase_Lotus.yaml')
    test = ObsProcessorPtv3(config)

    with open('/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_demo.pkl', 'rb') as f:
        demo = pickle.load(f)
    out = test.demo_2_obs_raw(demo)
    print(out.keys())
    test_rgb = out['rgb'][0][0]
    rbg_numpy = np.array(test_rgb)

    plt.imshow(rbg_numpy)
    plt.show()
    if record_example is True:
        with open('/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_raw_data.pkl', 'wb') as f:
            pickle.dump(out, f)


def test_preprocess(record_example=False):
    from zero.expAugmentation.config.default import get_config
    import matplotlib.pyplot as plt
    import pickle
    with open('/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_raw_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    config = get_config('/media/jian/ssd4t/zero/zero/expAugmentation/config/expBase_Lotus.yaml')
    test = ObsProcessorPtv3(config)
    out = test.static_process_fk(raw_data, '/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_episode')
    print(out.keys())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out['xyz'][0])
    pcd.colors = o3d.utility.Vector3dVector((out['rgb'][0] / 255))
    o3d.visualization.draw_geometries([pcd])

    if record_example is True:
        with open('/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_preprocess_data.pkl', 'wb') as f:
            pickle.dump(out, f)


def test_dynamic_process(record_example=False):
    from zero.expAugmentation.config.default import get_config
    import matplotlib.pyplot as plt
    import pickle
    with open('/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_preprocess_data.pkl', 'rb') as f:
        preprocess_data = pickle.load(f)

    config = get_config('/media/jian/ssd4t/zero/zero/expAugmentation/config/expBase_Lotus.yaml')
    test = ObsProcessorPtv3(config)
    out = test.dynamic_process(preprocess_data, 'close_jar_peract+0')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out['pc_fts'][0][:, :3])
    pcd.colors = o3d.utility.Vector3dVector((out['pc_fts'][0][:, 3:6] + 1) / 2)
    o3d.visualization.draw_geometries([pcd])
    print(out['theta_positions'])
    print(out.keys())


def test_inference():
    from zero.expAugmentation.config.default import get_config
    import pickle
    with open('/media/jian/ssd4t/zero/1_Data/C_Dataset_Example/example_obs.pkl', 'rb') as f:
        obs = pickle.load(f)
    config = get_config('/media/jian/ssd4t/zero/zero/expAugmentation/config/DP.yaml')
    obs_processor = ObsProcessorPtv3(config, train_flag=False)
    collect_fn = obs_processor.get_collect_function()

    obs_raw = obs_processor.obs_2_obs_raw(obs)
    obs_static = obs_processor.static_process_fk(obs_raw)
    obs_dynamic = obs_processor.dynamic_process(obs_static, 'close_jar_peract+0')
    batch = collect_fn([obs_dynamic])
    print(batch.keys())


# endregion


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def static_process():
    from zero.expAugmentation.config.default import get_config
    data_dir = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/20demo_put_groceries/train/500765'
    save_root = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/20demo_put_groceries/train/'
    config = get_config('/media/jian/ssd4t/zero/zero/expAugmentation/config/FK.yaml')
    check_and_make(os.path.join(save_root))

    tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
    obs_processor = ObsProcessorPtv3(config=config)
    obs_processor._dataset_init()
    for i, task in enumerate(tasks_all):
        variations = sorted(os.listdir(os.path.join(data_dir, task)), key=natural_sort_key)
        for j, variation in enumerate(variations):
            episodes = sorted(os.listdir(os.path.join(data_dir, task, variation, 'episodes')), key=natural_sort_key)
            for k, episode in tqdm(enumerate(episodes)):
                taskvar = f'{task}+{variation.split("variation")[-1]}'
                with open(os.path.join(data_dir, task, variation, 'episodes', episode, 'data.pkl'), 'rb') as f:
                    data = pickle.load(f)
                out = obs_processor.static_process_fk(data, taskvar)
                save_path = os.path.join(save_root, task, variation, episode)
                check_and_make(save_path)
                with open(os.path.join(save_path, 'data.pkl'), 'wb') as f:
                    pickle.dump(out, f)

    pass


if __name__ == '__main__':
    # test_collect_obs(record_example=True)
    # test_preprocess(record_example=True)
    # test_dataset_process(record_example=True)
    # test_inference()
    # test_dynamic_process()
    static_process()
