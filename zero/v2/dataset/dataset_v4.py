from collections.abc import Mapping, Container
from zero.v2.dataprocess.ObsProcessLotus import ObsProcessLotus
import sys
import pickle
import re
from zero.v2.tools_scripts.draw_pointcloud import PointCloudDrawer
import time
from zero.v2.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
from zero.v2.models.lotus.utils.robot_box import RobotBox
from zero.v2.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from zero.v2.config.constants import (
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


def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))


def pad_tensors(tensors, lens=None, pad=0, max_len=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens) if max_len is None else max_len
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


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


class SimplePolicyDataset(Dataset):
    def __init__(
        self, data_dir, instr_embed_file, taskvar_instr_file, taskvar_file=None,
        num_points=10000, xyz_shift='center', xyz_norm=True, use_height=False,
        rot_type='quat', instr_embed_type='last', all_step_in_batch=True,
        rm_table=True, rm_robot='none', include_last_step=False, augment_pc=False,
        sample_points_by_distance=False, same_npoints_per_example=False,
        rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
        pos_type='cont', pos_bins=50, pos_bin_size=0.01,
        pos_heatmap_type='plain', pos_heatmap_no_robot=False,
        aug_max_rot=45, real_robot=False, tasks_to_use=None, config=None, is_single_frame=True, **kwargs
    ):

        # 0. Parameters
        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']
        self.num_points = num_points
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.pos_type = pos_type
        self.rot_type = rot_type
        self.rm_table = rm_table
        self.rm_robot = rm_robot
        self.all_step_in_batch = all_step_in_batch
        self.include_last_step = include_last_step
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.sample_points_by_distance = sample_points_by_distance
        self.rm_pc_outliers = rm_pc_outliers
        self.rm_pc_outliers_neighbors = rm_pc_outliers_neighbors
        self.same_npoints_per_example = same_npoints_per_example
        self.euler_resolution = euler_resolution
        self.pos_bins = pos_bins
        self.pos_bin_size = pos_bin_size
        self.pos_heatmap_type = pos_heatmap_type
        self.pos_heatmap_no_robot = pos_heatmap_no_robot
        self.real_robot = real_robot
        self.is_single_frame = is_single_frame
        # 0.1. Load some pheripheral information
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()

        self.taskvar_instrs = json.load(open(taskvar_instr_file))
        self.instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}

        tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
        if tasks_to_use is not None:
            tasks_all = [task for task in tasks_all if task in tasks_to_use]
            print(f"tasks_all: {tasks_all}")

        # 1. episodes-wise list
        self.g_episode_to_taskvar = []  # Which taskvar is each episode
        self.g_episode_to_path = []  # retrieve all episodes path and put them in self.episodes
        self.g_episode_to_l_episode = []  # Which episode in each taskvar
        self.frames = []  # How many frames in each episode
        for task_name in tasks_all:
            task_folder_path = os.path.join(data_dir, task_name)
            variation_list = sorted(os.listdir(task_folder_path), key=natural_sort_key)
            for variation_folder in variation_list:
                l_episode = 0
                variation_folder_path = os.path.join(task_folder_path, variation_folder, 'episodes')
                episodes_list = sorted(os.listdir(variation_folder_path), key=natural_sort_key)
                for episode_folder in episodes_list:
                    episode_folder_path = os.path.join(variation_folder_path, episode_folder)
                    self.g_episode_to_path.append(episode_folder_path)
                    variation_id = int(variation_folder.split('variation')[-1])
                    taskvar = task_name + '_peract' + '+' + str(variation_id)
                    self.g_episode_to_taskvar.append(taskvar)
                    with open(os.path.join(episode_folder_path, 'data.pkl'), 'rb') as f:
                        data = pickle.load(f)
                    self.frames.append(len(data['data_ids']))

                    self.g_episode_to_l_episode.append(l_episode)
                    l_episode += 1

        # 2. frame-wise list
        self.g_frame_to_taskvar = []
        self.g_frame_to_g_episode = []
        self.g_frame_to_frame = []
        self.g_frame_to_l_episode = []

        for episode_id, frame in enumerate(self.frames):
            self.g_frame_to_g_episode.extend([episode_id] * frame)
            self.g_frame_to_taskvar.extend([self.g_episode_to_taskvar[episode_id]] * frame)
            self.g_frame_to_frame.extend(list(range(frame)))
            self.g_frame_to_l_episode.extend([episode_id] * frame)

        # 3.container
        self.cache = dict()
        self.max_cache_length = 1800
        print(f"max_cache_length: {self.max_cache_length}")
        for i in range(len(self.g_episode_to_path)):
            self.check_cache(i)

        self.op = ObsProcessLotus(config)
        self.config = config

    def check_cache(self, g_episode):

        if self.cache.get(g_episode) is None:

            episode_path = self.g_episode_to_path[g_episode]
            with open(os.path.join(episode_path, 'data.pkl'), 'rb') as f:
                data = pickle.load(f)

            if len(self.cache) >= self.max_cache_length:
                first_key = next(iter(self.cache))
                self.cache.pop(first_key)

            self.cache[g_episode] = data
            return data
        else:
            return self.cache[g_episode]

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

    def _augment_pc(self, xyz, action_current, action_next, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        action_current[:3] = random_rotate_z(action_current[:3], angle=angle)
        action_current[3:-1] = self._rotate_gripper(action_current[3:-1], angle)

        action_next[:3] = random_rotate_z(action_next[:3], angle=angle)
        action_next[3:-1] = self._rotate_gripper(action_next[3:-1], angle)

        if self.rot_type == 'quat':
            gt_rot = action_next[3:-1]
        elif self.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(action_next[3:-1][None, :]))[0].numpy() / 180.
        elif self.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(action_next[3:-1], self.euler_resolution)
        elif self.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(action_next[3:-1][None, :]))[0].numpy()

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, action_current, action_next, gt_rot

    def __exit__(self):
        for lmdb_env in self.lmdb_envs.values():
            lmdb_env.close()

    def __len__(self):
        if self.is_single_frame:
            return sum(self.frames)
        else:
            return len(self.frames)

    def __getitem__(self, g_frame_idx):
        # 0. get single frame
        if self.is_single_frame:
            return self.get_single_frame(g_frame_idx)
        else:
            return self.get_entire_episode(g_frame_idx)

    def get_entire_episode(self, g_idx):
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

        # 0.1 identify the frame info and output info
        taskvar = self.g_frame_to_taskvar[g_idx]
        g_episode = self.g_frame_to_g_episode[g_idx]
        l_episode = self.g_episode_to_l_episode[g_episode]
        frame_idx = self.g_frame_to_frame[g_idx]

        # 0.2 get data of specific frame
        data = self.check_cache(g_episode)
        num_frames = len(data['data_ids'])

        # 1.get specific frame data
        for t in range(num_frames):
            data_ids = data['data_ids'][t]
            xyz = copy.deepcopy(data['xyz'][t])
            rgb = copy.deepcopy(data['rgb'][t])
            action_current = copy.deepcopy(data['action_current'][t])
            action_next = copy.deepcopy(data['action_next'][t])
            arm_links_info = copy.deepcopy(data['arm_links_info'][t])
            instr_embed = copy.deepcopy(self.instr_embeds[random.choice(self.taskvar_instrs[taskvar])])

            # preprocess have done the following:
            # 1. voxelization
            # 2. remove outside workspace points
            # 3. remove table points
            # over

            # 4. remove robot points
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.config.TRAIN_DATASET.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

            # 5. downsampling
            if len(xyz) > self.config.TRAIN_DATASET.num_points:
                if self.config.TRAIN_DATASET.sample_points_by_distance:
                    dists = np.sqrt(np.sum((xyz - action_current[:3])**2, 1))
                    probs = 1 / np.maximum(dists, 0.1)
                    probs = np.maximum(softmax(probs), 1e-30)
                    probs = probs / sum(probs)
                    # probs = 1 / dists
                    # probs = probs / np.sum(probs)
                    point_idxs = np.random.choice(len(xyz), self.config.TRAIN_DATASET.num_points, replace=False, p=probs)
                else:
                    point_idxs = np.random.choice(len(xyz), self.config.TRAIN_DATASET.num_points, replace=False)
            else:
                if self.config.TRAIN_DATASET.same_npoints_per_example:
                    point_idxs = np.random.choice(xyz.shape[0], self.config.TRAIN_DATASET.num_points, replace=True)
                else:
                    point_idxs = np.arange(xyz.shape[0])
            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]
            # 6. remove outliers
            # /end of 6
            height = xyz[:, -1] - self.TABLE_HEIGHT

            robot_box = RobotBox(
                arm_links_info=arm_links_info,
                env_name='rlbench', selfgen=True
            )
            robot_point_idxs = np.array(
                list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
            )
            # 7. augment pc
            xyz, action_current, action_next, gt_rot = self._augment_pc(
                xyz, action_current, action_next,
                self.aug_max_rot,)

            # 8.normalize
            if self.config.TRAIN_DATASET.xyz_shift == 'none':
                centroid = np.zeros((3, ))
            elif self.config.TRAIN_DATASET.xyz_shift == 'center':
                centroid = np.mean(xyz, 0)
            elif self.config.TRAIN_DATASET.xyz_shift == 'gripper':
                centroid = copy.deepcopy(action_current[:3])

            if self.config.TRAIN_DATASET.xyz_norm:
                radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
            else:
                radius = 1

            xyz = (xyz - centroid) / radius
            height = height / radius
            action_current[:3] = (action_current[:3] - centroid) / radius
            action_next[:3] = (action_next[:3] - centroid) / radius
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)

            action_next = np.concatenate([action_next[:3], gt_rot, action_next[-1:]], 0)

            rgb = (rgb / 255) * 2 - 1
            pc_ft = np.concatenate([xyz, rgb], 1)
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

            if self.pos_type == 'disc':
                # (npoints, 3, 100)
                disc_pos_prob = get_disc_gt_pos_prob(
                    xyz, action_next[:3], pos_bins=self.pos_bins,
                    pos_bin_size=self.pos_bin_size,
                    heatmap_type=self.pos_heatmap_type,
                    robot_point_idxs=robot_point_idxs
                )
                outs['disc_pos_probs'].append(torch.from_numpy(disc_pos_prob).to(torch.float32))

            outs['data_ids'].append(data_ids)
            outs['pc_fts'].append(torch.from_numpy(pc_ft).to(torch.float32))
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).to(torch.float32))
            outs['ee_poses'].append(torch.from_numpy(action_current).to(torch.float32))
            outs['gt_actions'].append(torch.from_numpy(action_next).to(torch.float32))
            outs['step_ids'].append(t)
        return outs

    def get_single_frame(self, g_idx):
        # 0. get single frame
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

        # 0.1 identify the frame info and output info
        taskvar = self.g_frame_to_taskvar[g_idx]
        g_episode = self.g_frame_to_g_episode[g_idx]
        l_episode = self.g_episode_to_l_episode[g_episode]
        frame_idx = self.g_frame_to_frame[g_idx]

        # 0.2 get data of specific frame
        data = self.check_cache(g_episode)
        num_frames = len(data['data_ids'])
        t = frame_idx
        if frame_idx == num_frames:  # 最后一帧已经被去掉了
            t = frame_idx - 1

        # 1.get specific frame data
        outs['data_ids'].append(data['data_ids'][t])
        outs['step_ids'].append(data['step_ids'][t])
        outs['pc_fts'].append(data['pc_fts'][t])
        outs['ee_poses'].append(data['ee_poses'][t])
        outs['txt_embeds'].append(data['txt_embeds'][t])
        outs['gt_actions'].append(data['gt_actions'][t])
        outs['disc_pos_probs'].append(data['disc_pos_probs'][t])
        outs['pc_centroids'].append(data['pc_centroids'][t])
        outs['pc_radius'].append(data['pc_radius'][t])
        # print('1')
        # print(t)
        return outs


def base_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])

    for key in ['pc_fts', 'ee_poses', 'gt_actions']:
        batch[key] = torch.stack(batch[key], 0)

    batch['step_ids'] = torch.LongTensor(batch['step_ids'])

    txt_lens = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_masks'] = torch.from_numpy(
        gen_seq_masks(txt_lens, max_len=max(txt_lens))
    ).bool()
    batch['txt_embeds'] = pad_tensors(
        batch['txt_embeds'], lens=txt_lens, max_len=max(txt_lens)
    )

    # if len(batch['pc_centroids']) > 0:
    #     batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)
    #     batch['pc_radius'] = np.array(batch['pc_radius'])

    return batch


def ptv3_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])

    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)  # (#all points, 6)

    for key in ['ee_poses', 'gt_actions']:
        batch[key] = torch.stack(batch[key], 0)

    # if 'disc_pos_probs' in batch:
    #     batch['disc_pos_probs'] = batch['disc_pos_probs'] # [(3, #all pointspos_bins*2)]

    batch['step_ids'] = torch.LongTensor(batch['step_ids'])

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    return batch


if __name__ == '__main__':
    import yacs.config
    from tqdm import trange
    # seed everything
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config = yacs.config.CfgNode(new_allowed=True)
    config.merge_from_file('/workspace/zero/zero/v2/config/after_shock.yaml')
    # config.TRAIN_DATASET.tasks_to_use = ['close_jar']
    dataset = SimplePolicyDataset(config=config, is_single_frame=False, **config.TRAIN_DATASET)

    data = []
    for i in range(len(dataset)):
        data.append(dataset[i])
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
