import einops
import copy
from scipy.spatial.transform import Rotation as R
import torch
import random
from zero.v1.models.lotus.utils.robot_box import RobotBox
import open3d as o3d
import numpy as np


def process_pc(xyz, rgb, arm_links_info, voxel_size, visualize=False):
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


def dataset_part_process(data, ee_pose):
    num_points = 8192
    TABLE_HEIGHT = 0.7505

    xyz, rgb = data['xyz'].copy(), data['rgb'].copy()

    # print(f"xyz: {xyz.shape}")

    if len(xyz) > num_points:
        point_idxs = np.random.choice(len(xyz), num_points, replace=False)
    else:
        max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
        point_idxs = np.random.permutation(len(xyz))[:max_npoints]

    xyz = xyz[point_idxs]
    rgb = rgb[point_idxs]
    height = xyz[:, -1] - TABLE_HEIGHT

    # point cloud augmentation

    # normalize point cloud

    centroid = np.mean(xyz, 0)

    radius = 1
    xyz = (xyz - centroid) / radius
    height = height / radius
    rgb = (rgb / 255.) * 2 - 1
    pc_ft = np.concatenate([xyz, rgb], 1)
    ee_pose[:3] = (ee_pose[:3] - centroid) / radius
    pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

    return pc_ft, centroid, radius
