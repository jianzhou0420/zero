import argparse
from zero.v1.models.lotus.utils.robot_box import RobotBox
import open3d as o3d
import pickle

from tqdm import tqdm
import lmdb
import os
import msgpack
# Path to your LMDB database
import re
from codebase.Tools.PointCloudDrawer import PointCloudDrawer
import numpy as np
import matplotlib.pyplot as plt
pcdrawer = PointCloudDrawer()


def get_mask_with_robot_box(xyz, arm_links_info, rm_robot_type):
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


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def retrieve_all_episodes(root_path):

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


def process_pc(xyz, rgb, arm_links_info, voxel_size, visualize=False):

    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)

    # remove robot
    mask = get_mask_with_robot_box(xyz, arm_links_info=arm_links_info, rm_robot_type='box_keep_gripper')
    xyz = xyz[mask]
    rgb = rgb[mask]
    # remove table
    TABLE_HEIGHT = 0.7505
    mask = xyz[:, 2] > TABLE_HEIGHT
    xyz = xyz[mask]
    rgb = rgb[mask]

    # remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
    pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
    xyz = xyz[outlier_masks]
    rgb = rgb[outlier_masks]

    # print('xyz:', xyz.shape, 'rgb:', rgb.shape)
    # voxelization
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    xyz = []
    rgb = []
    for voxel in voxel_grid.get_voxels():
        # Get the center of the voxel
        xyz.append(voxel.grid_index * voxel_grid.voxel_size + voxel_grid.origin)
        rgb.append(voxel.color)

    # Visualize the mesh
    if visualize:
        o3d.visualization.draw_geometries([voxel_grid])
    xyz = np.array(xyz)
    rgb = np.array(rgb)
    return xyz, rgb


argparser = argparse.ArgumentParser()
argparser.add_argument('--voxel_size', type=float, default=0.001, required=True)
args = argparser.parse_args()

# region test
tasks_list = retrieve_all_episodes("/media/jian/ssd4t/selfgen/20250105/train_dataset/keysteps/seed42")
output_dir = f'/media/jian/ssd4t/selfgen/20250105/train_dataset/post_process_keysteps/seed42/voxel{args.voxel_size}'


# nested sum
total = sum([len(variation) for task in tasks_list for variation in task])
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
            new_pc_list = []
            new_rgb_list = []
            for frame_id in range(rgb.shape[0]):
                arm_links_info = (data['bbox'][frame_id], data['pose'][frame_id])
                new_xyz, new_rgb = process_pc(xyz[frame_id], rgb[frame_id], arm_links_info, voxel_size=args.voxel_size, visualize=False)
                new_pc_list.append(new_xyz)
                new_rgb_list.append(new_rgb)
            data['pc'] = new_pc_list
            data['rgb'] = new_rgb_list
            sub = data_path.split('/seed42')
            export_path = os.path.join(output_dir, sub[1][1:])
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, 'wb') as f:
                pickle.dump(data, f)
            pbar.update(1)
# endregion
