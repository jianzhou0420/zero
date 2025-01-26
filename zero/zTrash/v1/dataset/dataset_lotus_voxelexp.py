import pickle
import re
from zero.v1.tools_scripts.draw_pointcloud import PointCloudDrawer
import time
from zero.v1.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
from zero.v1.models.lotus.utils.robot_box import RobotBox
from zero.v1.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from zero.v1.config.constants import (
    get_rlbench_labels, get_robot_workspace
)

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import json
import copy
import random
from scipy.special import softmax

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

# import open3d as o3d
""""""


def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class test(Dataset):
    def __init__(self,
                 data_dir='/media/jian/ssd4t/selfgen/20250105/train_dataset/post_process_keysteps/seed42/voxel0.005',
                 tasks_to_use=None,
                 ):
        # TODO:Complete the parameters
        taskvar_instr_file = '/data/zero/zero/v1/models/lotus/assets/taskvars_instructions_peract.json'
        instr_embed_file = '/data/lotus/peract/train/keysteps_bbox_pcd/instr_embeds_clip.npy'
        self.rot_type = 'euler_disc'
        self.euler_resolution = 5
        self.taskvar_instrs = json.load(open(taskvar_instr_file))
        self.instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        self.TABLE_HEIGHT = 0.7505
        # 1. Retrieve all episodes,path
        # task_dict(variation_list(episodes_list(path)))
        all_tasks = sorted(os.listdir(data_dir), key=natural_sort_key)
        all_tasks_dict = {}
        for task in all_tasks:
            single_task_path = os.path.join(data_dir, task)
            single_task_all_variations = sorted(os.listdir(single_task_path), key=natural_sort_key)
            single_tasks_all_variations_list = []
            for variation in single_task_all_variations:
                single_variation_path = os.path.join(single_task_path, variation, 'episodes')
                single_variation_all_episodes = sorted(os.listdir(single_variation_path), key=natural_sort_key)
                single_variation_all_episodes_list = []
                for episode in single_variation_all_episodes:
                    single_episode_path = os.path.join(single_variation_path, episode)
                    single_variation_all_episodes_list.append(single_episode_path)
                single_tasks_all_variations_list.append(single_variation_all_episodes_list)
            all_tasks_dict[task] = single_tasks_all_variations_list

        if tasks_to_use is not None:
            self.tasks_to_use = tasks_to_use
            self.tasks_to_use_dict = {key: all_tasks_dict[key] for key in tasks_to_use if key in all_tasks_dict}
        else:
            self.tasks_to_use = all_tasks
            self.tasks_to_use_dict = all_tasks_dict
        # 2. Cache all the episodes
        # cache all the episodes and load entire episodes for retrieving one frame
        self.cache = []

        self.tasks_in_episodes = []
        for taskname, single_task_variation_list in self.tasks_to_use_dict.items():
            for variation_id, single_variation_episode_list in enumerate(single_task_variation_list):
                for single_episode_path in single_variation_episode_list:
                    with open(os.path.join(single_episode_path, 'data.pkl'), 'rb') as f:
                        data = pickle.load(f)
                    taskvar = taskname + '_peract' + '+' + str(variation_id)
                    self.tasks_in_episodes.append(taskvar)
                    self.cache.append(data)

        # 3. idx translation
        self.globalframe_to_taskvar = []
        self.globalframe_to_episode = []
        self.globanframe_to_frame = []

        self.frames = [len(data['key_frameids']) for data in self.cache]
        self.lenth = sum(self.frames)

        for i, frame in enumerate(self.frames):
            self.globalframe_to_episode.extend([i] * frame)
            self.globalframe_to_taskvar.extend([self.tasks_in_episodes[i]] * frame)
        for i, frame in enumerate(self.frames):
            self.globanframe_to_frame.extend(list(range(frame)))

        print(len(self.globalframe_to_taskvar), len(self.globalframe_to_episode), len(self.globanframe_to_frame))

    def __len__(self):
        return self.lenth

    def get_groundtruth_rotations(self, ee_poses):
        gt_rots = torch.from_numpy(ee_poses.copy())   # quaternions
        if self.rot_type == 'euler':    # [-1, 1]
            gt_rots = self.rotation_transform.quaternion_to_euler(gt_rots[1:]) / 180.
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        elif self.rot_type == 'euler_disc':  # 3D
            gt_rots = [quaternion_to_discrete_euler(x, self.euler_resolution) for x in gt_rots[1:]]
            gt_rots = torch.from_numpy(np.stack(gt_rots + gt_rots[-1:]))
        elif self.rot_type == 'euler_delta':
            gt_eulers = self.rotation_transform.quaternion_to_euler(gt_rots)
            gt_rots = (gt_eulers[1:] - gt_eulers[:-1]) % 360
            gt_rots[gt_rots > 180] -= 360
            gt_rots = gt_rots / 180.
            gt_rots = torch.cat([gt_rots, torch.zeros(1, 3)], 0)
        elif self.rot_type == 'rot6d':
            gt_rots = self.rotation_transform.quaternion_to_ortho6d(gt_rots)
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        else:
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        gt_rots = gt_rots.numpy()
        return gt_rots

    def __getitem__(self, global_frame_idx):
        self.pos_heatmap_type = 'dist'
        self.pos_bins = 15
        self.pos_bin_size = 0.01
        self.pos_heatmap_no_robot = True
        self.augment_pc = True
        self.xyz_shift = 'center'
        self.xyz_norm = False
        self.use_height = True
        self.pos_type = 'disc'
        self.aug_max_rot = np.deg2rad(45)
        #
        taskvar = self.globalframe_to_taskvar[global_frame_idx]
        episodes_idx = self.globalframe_to_episode[global_frame_idx]
        frame_idx = self.globanframe_to_frame[global_frame_idx]
        data = self.cache[episodes_idx]
        outs = {
            'data_ids': [], 'pc_fts': [], 'step_ids': [],
            'pc_centroids': [], 'pc_radius': [], 'ee_poses': [],
            'txt_embeds': [], 'gt_actions': [],
        }

        if self.pos_type == 'disc':
            outs['disc_pos_probs'] = []

        gt_rots = self.get_groundtruth_rotations(data['action'][:, 3:7])

        num_steps = len(data['pc'])
        t = frame_idx

        if t == num_steps - 1:  # 因为我在self.frames里面没有算最后一帧，所以这里就不会选中最后一帧, 属于bug与特殊需求相互抵消了，哈哈哈哈哈哈
            t -= 1
            print(f"t: {t}")

        xyz, rgb = data['pc'][t], data['rgb'][t]

        arm_links_info = (data['bbox'][t], data['pose'][t])

        gt_action = copy.deepcopy(data['action'][t + 1])
        current_pose = copy.deepcopy(data['action'][t])
        gt_rot = gt_rots[t]

        # randomly select one instruction
        instr = random.choice(self.taskvar_instrs[taskvar])
        instr_embed = self.instr_embeds[instr]

        height = xyz[:, -1] - self.TABLE_HEIGHT

        if self.pos_heatmap_no_robot:
            robot_box = RobotBox(
                arm_links_info=arm_links_info,
                env_name='rlbench'
            )
            robot_point_idxs = np.array(
                list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
            )
        else:
            robot_point_idxs = None

        # point cloud augmentation
        if self.augment_pc:
            xyz, current_pose, gt_action, gt_rot = self._augment_pc(
                xyz, current_pose, gt_action, gt_rot, self.aug_max_rot
            )

        # normalize point cloud
        if self.xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif self.xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif self.xyz_shift == 'gripper':
            centroid = copy.deepcopy(current_pose[:3])

        if self.xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        gt_action[:3] = (gt_action[:3] - centroid) / radius
        current_pose[:3] = (current_pose[:3] - centroid) / radius
        outs['pc_centroids'].append(centroid)
        outs['pc_radius'].append(radius)

        gt_action = np.concatenate([gt_action[:3], gt_rot, gt_action[-1:]], 0)

        rgb = (rgb / 255.) * 2 - 1
        pc_ft = np.concatenate([xyz, rgb], 1)
        if self.use_height:
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

        if self.pos_type == 'disc':
            # (npoints, 3, 100)
            disc_pos_prob = get_disc_gt_pos_prob(
                xyz, gt_action[:3], pos_bins=self.pos_bins,
                pos_bin_size=self.pos_bin_size,
                heatmap_type=self.pos_heatmap_type,
                robot_point_idxs=robot_point_idxs
            )
            outs['disc_pos_probs'].append(torch.from_numpy(disc_pos_prob))

        outs['data_ids'].append(f'{taskvar}-{episodes_idx}-t{t}')
        outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
        outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
        outs['ee_poses'].append(torch.from_numpy(current_pose).float())
        outs['gt_actions'].append(torch.from_numpy(gt_action).float())
        outs['step_ids'].append(t)

        return outs

    def _augment_pc(self, xyz, ee_pose, gt_action, gt_rot, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        ee_pose[:3] = random_rotate_z(ee_pose[:3], angle=angle)
        gt_action[:3] = random_rotate_z(gt_action[:3], angle=angle)
        ee_pose[3:-1] = self._rotate_gripper(ee_pose[3:-1], angle)
        gt_action[3:-1] = self._rotate_gripper(gt_action[3:-1], angle)
        if self.rot_type == 'quat':
            gt_rot = gt_action[3:-1]
        elif self.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy() / 180.
        elif self.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(gt_action[3:-1], self.euler_resolution)
        elif self.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy()

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, ee_pose, gt_action, gt_rot

    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot


if __name__ == '__main__':
    dataset = test()
    dataset[0]
