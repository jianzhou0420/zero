import pickle
import re
# from ..models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
# from ..models.lotus.utils.robot_box import RobotBox
# from ..models.lotus.utils.rotation_transform import (
#     RotationMatrixTransform, quaternion_to_discrete_euler
# )
# from ..config.constants import (
#     get_rlbench_labels, get_robot_workspace
# )

from zero.expForwardKinematics.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
from zero.expForwardKinematics.models.lotus.utils.robot_box import RobotBox
from zero.expForwardKinematics.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from zero.expForwardKinematics.config.constants import (
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
from zero.expForwardKinematics.ObsProcessor.ObsProcessorPtv3 import ObsProcessorPtv3

# region utils


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

# endregion

# region dataset


class LotusDatasetAugmentation(Dataset):
    def __init__(
        self, instr_embed_file, taskvar_instr_file,
        num_points=10000000, xyz_shift='center', xyz_norm=True, use_height=False,
        rot_type='quat', instr_embed_type='last',
        rm_robot='none', augment_pc=False,
        euler_resolution=5,
        pos_type='cont', pos_bins=50, pos_bin_size=0.01,
        pos_heatmap_type='plain', pos_heatmap_no_robot=False,
        aug_max_rot=45, real_robot=False, tasks_to_use=None, config=None, data_dir=None,
        **kwargs
    ):
        # 0. Parameters
        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
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
        self.rot_type = rot_type
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.euler_resolution = euler_resolution

        self.pos_type = pos_type
        self.pos_bins = pos_bins
        self.pos_bin_size = pos_bin_size
        self.pos_heatmap_type = pos_heatmap_type
        self.pos_heatmap_no_robot = pos_heatmap_no_robot

        # 0.1. Load some pheripheral information
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()

        self.config = config
        data_dir = data_dir
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

        # perceptor
        self.ptv3_perceptor = ObsProcessorPtv3(config)

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
        # pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        # xyz = pc_noises + xyz

        return xyz, action_current, action_next, gt_rot

    def __exit__(self):
        for lmdb_env in self.lmdb_envs.values():
            lmdb_env.close()

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, g_episode):
        # 0. get single frame
        return self.get_entire_episode(g_episode)

    def new_dataset(self, g_episode):
        taskvar = self.g_episode_to_taskvar[g_episode]
        data = self.check_cache(g_episode)
        outs = self.ptv3_perceptor.dynamic_process(data, taskvar)
        return outs

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
            'disc_pos_probs': []
        }

        # 0.1 identify the frame info and output info
        taskvar = self.g_episode_to_taskvar[g_episode]
        # print(f"taskvar: {taskvar}")
        # 0.2 get data of specific frame
        data = self.check_cache(g_episode)

        num_frames = len(data['data_ids'])

        # 1.get specific frame data
        for t in range(num_frames):
            data_ids = data['data_ids'][t]
            xyz = copy.deepcopy(data['xyz'][t])
            rgb = copy.deepcopy(data['rgb'][t])
            ee_pose = copy.deepcopy(data['action_current'][t])
            gt_action = copy.deepcopy(data['action_next'][t])
            arm_links_info = copy.deepcopy(data['arm_links_info'][t])

            xyz, rgb = data['xyz'][t], data['rgb'][t]

            # randomly select one instruction
            instr = random.choice(self.taskvar_instrs[taskvar])
            instr_embed = copy.deepcopy(self.instr_embeds[instr])

            # 5. downsample point cloud
            if len(xyz) > self.num_points:
                point_idxs = np.random.choice(len(xyz), self.num_points, replace=False)
                xyz = xyz[point_idxs]
                rgb = rgb[point_idxs]
            max_npoints = int(len(xyz) * np.random.uniform(0.4, 0.6))
            point_idxs = np.random.permutation(len(xyz))[:max_npoints]
            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]
            height = xyz[:, -1] - self.TABLE_HEIGHT
            # print(f"After downsample xyz: {xyz.shape}")

            if self.pos_heatmap_no_robot:
                robot_box = RobotBox(
                    arm_links_info=arm_links_info,
                    env_name='rlbench', selfgen=True
                )
                robot_point_idxs = np.array(
                    list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
                )
            else:
                robot_point_idxs = None

            # 6. point cloud augmentation
            if self.augment_pc:
                xyz, ee_pose, gt_action, gt_rot = self._augment_pc(
                    xyz, ee_pose, gt_action, self.aug_max_rot
                )

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
            gt_action[:3] = (gt_action[:3] - centroid) / radius
            ee_pose[:3] = (ee_pose[:3] - centroid) / radius
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

            # print(f"{taskvar}: {xyz.shape}")
            outs['data_ids'].append(data_ids)
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_actions'].append(torch.from_numpy(gt_action).float())
            outs['step_ids'].append(t)
            # break
        return outs
# endregion
# region methods


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
# endregion


if __name__ == '__main__':
    import argparse
    import yacs.config
    from tqdm import trange
    from zero.expForwardKinematics.config.default import build_args
    config = build_args()

    train_data_path = os.path.join(config.B_Preprocess, 'train')
    dataset = LotusDatasetAugmentation(config=config, tasks_to_use=config.tasks_to_use, data_dir=train_data_path, **config.TRAIN_DATASET)

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

    # with open('/data/zero/assets/peract_tasks.json', 'r') as f:
    #     tasks = json.load(f)

    # for i in range(18):

    #     single_frame = dataset[i * 100]

    #     print(f"task: {single_frame['data_ids'][0].split('-')[1]:<35}", single_frame['pc_fts'][0].shape[0])
    # dataset[i]、
    print(f"len(dataset): {len(dataset)}")
    # break
    '''
     python  -m zero.expForwardKinematics.dataset.dataset_expbase_voxel_augment \
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
            B_Preprocess /data/zero/1_Data/B_Preprocess/0.005all_with_path_voxelcenter \
            des "to see close_jar and insert at 0.005"\
    '''
