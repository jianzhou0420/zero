import einops
import pickle
import re
from ..models.lotus.utils.robot_box import RobotBox
from ..models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from ..config.constants import (
    get_rlbench_labels, get_robot_workspace
)
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import json
import copy
import random


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


class LotusDatasetAugmentation(Dataset):
    def __init__(
        self, instr_embed_file, taskvar_instr_file,
        num_points=10000000, xyz_shift='center', xyz_norm=True, use_height=False,
        rot_type='quat', instr_embed_type='last',
        rm_robot='none', augment_pc=False,
        euler_resolution=5,
        aug_max_rot=45, real_robot=False, tasks_to_use=None, config=None, data_dir=None,
        **kwargs
    ):
        # 0. Parameters
        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']

        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']

        # 0.1 Downsample args
        self.num_points = num_points

        # 0.2 shift and normalization
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm

        # put together
        self.use_height = use_height

        # augment & action head
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.euler_resolution = euler_resolution

        # 0.1. Load some pheripheral information
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()

        self.config = config
        data_dir = data_dir
        self.taskvar_instrs = json.load(open(taskvar_instr_file))
        self.instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()
        if instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}

        self.variations_to_use = self.config.TRAIN_DATASET.variations_to_use

        tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
        if tasks_to_use is not None:
            tasks_all = [task for task in tasks_all if task in tasks_to_use]
        if len(tasks_all) == 1:
            can_use_variation_flag = True
        else:
            can_use_variation_flag = False
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

                if can_use_variation_flag == True and self.variations_to_use is not None:
                    if int(variation_folder.split('variation')[-1]) not in self.variations_to_use:
                        print(f"Skip {variation_folder}")
                        continue
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

        new_action_next = []
        for i, action in enumerate(action_next):
            action[:3] = random_rotate_z(action[:3], angle=angle)
            action[3:-1] = self._rotate_gripper(action[3:-1], angle)
            new_action_next.append(action)
        action_next = np.stack(new_action_next, 0)

        gt_rot = []
        for action in action_next:
            gt_rot.append(quaternion_to_discrete_euler(action[3:-1], self.euler_resolution))
        gt_rot = np.stack(gt_rot, 0)

        # add small noises (+-2mm)
        # pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        # xyz = pc_noises + xyz

        return xyz, action_current, action_next, gt_rot

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, g_episode):
        # 0. get single frame
        return self.get_entire_episode(g_episode)

    def _find_gt_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg'):
        if sub_keyframe_dection_mode == 'avg':

            indices = np.linspace(0, len(actions_path) - 1, self.config.horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
            gt_actions = [actions_path[i] for i in indices]
            gt_theta_actions = [theta_actions_path[i] for i in indices]
            return gt_actions, gt_theta_actions
        elif sub_keyframe_dection_mode == 'xyzpeak':
            NotImplementedError("XYZPEAK")

    def get_entire_episode(self, g_episode):
        '''
        主要就做augmentation的工作
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
            'disc_pos_probs': [],
            'theta_positions': []
        }

        # 0.1 identify the frame info and output info
        taskvar = self.g_episode_to_taskvar[g_episode]
        # print(f"taskvar: {taskvar}")
        # 0.2 get data of specific frame
        data = self.check_cache(g_episode)
        num_frames = len(data['data_ids'])

        # 1.get specific frame data
        for t in range(num_frames):
            sub_keyframe_dection_mode = 'avg'
            assert sub_keyframe_dection_mode in ['avg', 'xyzpeak']

            # end of path processs
            data_ids = data['data_ids'][t]
            xyz = copy.deepcopy(data['xyz'][t])
            rgb = copy.deepcopy(data['rgb'][t])
            ee_pose = copy.deepcopy(data['action_current'][t])
            action_next = copy.deepcopy(data['action_next'][t])
            action_path = copy.deepcopy(data['actions_path'][t])
            theta_actions_path = copy.deepcopy(data['theta_actions_path'][t])
            gt_actions, gt_theta_actions = self._find_gt_actions(action_path, theta_actions_path, sub_keyframe_dection_mode)
            # assert (gt_actions[0] == ee_pose).all()
            assert (gt_actions[-1] == action_next).all()
            assert len(gt_actions) == self.config.horizon

            # append open to theta_actions
            for i in range(len(gt_theta_actions)):
                gt_theta_actions[i] = np.append(gt_theta_actions[i], gt_actions[i][-1])

            xyz, rgb = data['xyz'][t], data['rgb'][t]

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
            if self.config.unit_test is False:
                if self.augment_pc:
                    xyz, ee_pose, gt_actions, gt_rot = self._augment_pc(
                        xyz, ee_pose, gt_actions, self.aug_max_rot
                    )
                if tmp_flag:
                    pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
                    xyz = pc_noises + xyz
            else:
                gt_rot = []
                for action in gt_actions:
                    gt_rot.append(quaternion_to_discrete_euler(action[3:-1], self.euler_resolution))
                gt_rot = np.stack(gt_rot, 0)
                gt_actions = np.stack(gt_actions, 0)

            # 7. normalize point cloud
            if self.xyz_shift == 'none':
                centroid = np.zeros((3, ))
            elif self.xyz_shift == 'center':
                centroid = np.mean(xyz, 0)
            elif self.xyz_shift == 'gripper':
                centroid = copy.deepcopy(ee_pose[:3])
            if self.xyz_norm:
                radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
            else:
                radius = 1

            xyz = (xyz - centroid) / radius
            height = height / radius
            gt_actions[:, :3] = (gt_actions[:, :3] - centroid) / radius
            ee_pose[:3] = (ee_pose[:3] - centroid) / radius
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)

            gt_actions = np.concatenate([gt_actions[:, :3], gt_rot, gt_actions[:, -1:]], 1)

            rgb = (rgb / 255.) * 2 - 1
            pc_ft = np.concatenate([xyz, rgb], 1)
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

            # print(f"{taskvar}: {xyz.shape}")
            outs['data_ids'].append(data_ids)
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_actions'].append(torch.from_numpy(gt_actions).float())
            outs['step_ids'].append(t)
            test = torch.from_numpy(np.array(gt_theta_actions)).float()
            test = einops.rearrange(test, 'h a -> a h')  # 现在channel是各个纬度的action
            outs['theta_positions'].append(torch.from_numpy(np.array(test)).float())
        #     print(gt_theta_actions)
        # with open('/data/zero/1_Data/C_Dataset_Example/example.pkl', 'wb') as f:
        #     pickle.dump(outs, f)
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

    for key in ['ee_poses', 'gt_actions', 'theta_positions']:
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
    import argparse
    import yacs.config
    from tqdm import trange
    from ..config.default import build_args
    config = build_args()

    dataset = LotusDatasetAugmentation(config=config, is_single_frame=False, tasks_to_use=config.tasks_to_use, data_dir=config.B_Preprocess, **config.TRAIN_DATASET)

    # all_data = []
    # for i in trange(len(dataset)):
    #     data = dataset[i]
    #     all_data.append(data)
    # with open(args.output, 'wb') as f:
    #     pickle.dump(all_data, f)
    # xyz_all=[]
    # single_xyz=[]
    # for i in trange(len(dataset)):
    #     data = dataset.check_cache(i)
    #     xyz = data['xyz']
    #     for j in range(len(xyz)):
    #         single_xyz.append(xyz[j].shape)
    #     if i % 100 == 0:
    #         single_xyz = np.average(single_xyz, axis=0)
    #         xyz_all.append(single_xyz)
    #         single_xyz = []

    print(f"len(dataset): {len(dataset)}")
    test = dataset[0]
    # dataset[i]
    # break
    '''
     python  -m zero.expForwardKinematics.dataset.dataset_expbase_DP \
            --exp-config /data/zero/zero/expForwardKinematics/config/expBase_Lotus.yaml \
            name EXP03_04_insert_close_jar_0.005\
            dataset augment\
            num_gpus 1 \
            epoches 800 \
            batch_size 4 \
            TRAIN_DATASET.num_points 4096 \
            TRAIN_DATASET.pos_bins 75 \
            TRAIN_DATASET.pos_bin_size 0.001 \
            MODEL.action_config.pos_bins 75 \
            MODEL.action_config.pos_bin_size 0.001 \
            MODEL.action_config.voxel_size 0.005\
            TRAIN.n_workers 4\
            B_Preprocess /data/zero/1_Data/B_Preprocess/0.005all_with_path_with_positionactions/train \
            tasks_to_use close_jar \
    '''
