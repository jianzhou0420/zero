import pickle
import torch
import numpy as np
import open3d as o3d

# region batch analysis
import argparse


import tap


class BatchAnalysis(tap.Tap):
    batch_file: str = None


args = BatchAnalysis().parse_args()


with open(args.batch_file, 'rb') as f:
    all_data = pickle.load(f)

print(all_data[0].keys())
# 我需要的数据
# 1. 点云数量


def single_episode_analysis(episode):
    data_ids = episode['data_ids']
    pc_fts = episode['pc_fts']
    pc_centroids = episode['pc_centroids']
    pc_radius = episode['pc_radius']
    ee_poses = episode['ee_poses']
    gt_actions = episode['gt_actions']

    num_pc_list = []
    for i in range(len(data_ids)):
        num_pc_list.append(pc_fts[i].shape[0])

    return num_pc_list


num_pc_list = []

for episode in all_data:
    num_pc_list.extend(single_episode_analysis(episode))

num_pc_list = np.array(num_pc_list)
larger_than_num_points = num_pc_list[num_pc_list > 10000]
print('mean:', num_pc_list.mean())
print('std', num_pc_list.std())
print('max', num_pc_list.max())
print('min', num_pc_list.min())
print('num larger than 10000:', len(larger_than_num_points))


# endregion
