import random
import json
import torch
from zero.v2.models.lotus.utils.action_position_utils import get_disc_gt_pos_prob
import yacs.config
from zero.v1.dataset.dataset_lotus_voxelexp_copy import SimplePolicyDataset
from tqdm import trange
import argparse

import open3d as o3d
import pickle

from tqdm import tqdm
import lmdb
import os
import msgpack
# Path to your LMDB database
import re

import numpy as np
import matplotlib.pyplot as plt
import copy
from zero.v2.dataprocess.ObsProcessLotus import ObsProcessLotus


from zero.v2.dataprocess.utils import natural_sort_key, get_mask_with_robot_box


class Voxelizer:
    def __init__(self):
        pass

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

    def process_Data(self, data, voxel_size, episode_path, instr_embeds, taskvar_instrs, visualize=False):
        op = ObsProcessLotus()
        all_names = episode_path.split('/')
        task_name = all_names[9]
        variation_name = all_names[10].split('variation')[-1]
        episode_name = all_names[12]
        taskvar = f'{task_name}_peract+{variation_name}'
        # print(rgb.shape, xyz.shape)
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
            'ee_poses': [],
            'txt_embeds': [],
            'gt_actions': [],
            'disc_pos_probs': []
        }

        for t in range(data['pc'].shape[0]):
            if t == data['pc'].shape[0] - 1:
                continue

            # input rgb xyz arm_links_info,config
            rgb = data['rgb'][t] / 255.0
            xyz = data['pc'][t]
            arm_links_info = (data['bbox'][t], data['pose'][t])

            xyz = xyz.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)

            # restrict to the robot workspace
            in_mask = op.cut_workspace(xyz)
            xyz, rgb = xyz[in_mask], rgb[in_mask]
            # remove irrelevant objects
            xyz, rgb = op.remove_robot(xyz, rgb, arm_links_info, rm_robot_type='box_keep_gripper')    # remove robot
            xyz, rgb = op.remove_table(xyz, rgb)    # remove table
            xyz, rgb = op.remove_outliers(xyz, rgb)  # remove outliers

            # voxelization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

            # center of the voxel
            xyz = []
            rgb = []
            for voxel in voxel_grid.get_voxels():
                xyz.append(voxel.grid_index * voxel_grid.voxel_size + voxel_grid.origin)
                rgb.append(voxel.color)
            xyz = np.array(xyz)
            rgb = np.array(rgb)

            # dataset part process

            action_current = copy.deepcopy(data['action'][t])
            action_next = copy.deepcopy(data['action'][t + 1])

            gt_rot = op.get_groundtruth_rotations(data['action'][:, 3:7])[t]

            # 1. some process
            # xyz, rgb = op.remove_table(xyz, rgb)
            # xyz, rgb = op.remove_robot(xyz, rgb, links_info)
            xyz, rgb = op.downsample(xyz, rgb, action_current, config.sample_points_by_distance, num_points=config.num_points,)

            # robot point idxs
            robot_point_idxs = op.get_robot_point_idxs(config.pos_heatmap_no_robot, xyz, arm_links_info)
            height = xyz[:, -1] - 0.7505
            # point cloud augmentation
            xyz, action_current, action_next, gt_rot = op.augment_pc(config.augment_pc, xyz, action_current, action_next, gt_rot, config.aug_max_rot, config.rot_type, config.euler_resolution)

            # normalize point cloud
            centroid, radius, height, xyz, action_current, action_next = op.normalize_pc(config.xyz_shift, config.xyz_norm, xyz, action_current, action_next, gt_rot, height)

            # process the rgb
            pc_ft = op.process_pc(xyz, rgb, height, config.use_height)

            # (npoints, 3, 100)
            disc_pos_prob = get_disc_gt_pos_prob(
                xyz, action_next[:3], pos_bins=config.pos_bins,
                pos_bin_size=config.pos_bin_size,
                heatmap_type=config.pos_heatmap_type,
                robot_point_idxs=robot_point_idxs
            )

            # randomly select one instruction
            instr_embed = instr_embeds[random.choice(taskvar_instrs[taskvar])]

            outs['disc_pos_probs'].append(torch.from_numpy(disc_pos_prob))
            outs['data_ids'].append(f'{task_name}-{variation_name}-{episode_name}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['txt_embeds'].append(torch.from_numpy(instr_embed).float())
            outs['ee_poses'].append(torch.from_numpy(action_current).float())
            outs['gt_actions'].append(torch.from_numpy(action_next).float())
            outs['step_ids'].append(t)
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)
        return outs


argparser = argparse.ArgumentParser()
argparser.add_argument('--voxel_size', type=float, default=0.005)
args = argparser.parse_args()


voxelizer = Voxelizer()
for voxel_size in [0.001, 0.002, 0.003]:
    args.voxel_size = voxel_size
    config = yacs.config.CfgNode(new_allowed=True)
    config.merge_from_file(f'/workspace/zero/zero/v2/config/lotus_{args.voxel_size}.yaml')
    config = config.TRAIN_DATASET

    # region test
    tasks_list = voxelizer.retrieve_all_episodes("/media/jian/ssd4t/selfgen/20250105/train_dataset/keysteps/seed42")
    output_dir = f'/media/jian/ssd4t/selfgen/test/test/seed42/voxel{args.voxel_size}'

    taskvar_instr_file = '/workspace/zero/zero/v2/models/lotus/assets/taskvars_instructions_peract.json'
    instr_embed_file = '/data/lotus/peract/train/keysteps_bbox_pcd/instr_embeds_clip.npy'

    taskvar_instrs = json.load(open(taskvar_instr_file))
    instr_embeds = np.load(instr_embed_file, allow_pickle=True).item()

    if config.instr_embed_type == 'last':
        instr_embeds = {instr: embeds[-1:] for instr, embeds in instr_embeds.items()}

    total = sum([len(variation) for task in tasks_list for variation in task])    # nested sum
    pbar = tqdm(total=total, desc=f"{args.voxel_size}")
    for task in tasks_list:
        # if 'insert_onto_square_peg' not in task[0][0]:
        #     continue
        for variation in task:
            for episode in variation:
                data_path = episode
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                # print(data.keys())
                rgb = data['rgb']
                xyz = data['pc']
                # print(rgb.shape, xyz.shape)
                new_data = voxelizer.process_Data(data, args.voxel_size, episode, instr_embeds, taskvar_instrs, visualize=False)
                sub = data_path.split('/seed42')
                export_path = os.path.join(output_dir, sub[1][1:])
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                with open(export_path, 'wb') as f:
                    pickle.dump(new_data, f)
                pbar.update(1)

# endregion
