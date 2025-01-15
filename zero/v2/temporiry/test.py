import copy
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import open3d

from zero.v2.models.lotus.utils.robot_box import RobotBox
from scipy.special import softmax
import torch

from zero.v2.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from sklearn.neighbors import LocalOutlierFactor
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


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


class ObsProcessLotus:
    '''
    Processor that convert raw pcd and rgb to voxelized pcd 

    Three stages Process:
    1. Pointlize: RGBD and Camera parameters (both intrincit and extrinsic) to raw pcd [xyz, rgb]
    2. Downsample: raw pcd [xyz, rgb] to voxelized pcd
    3. Trim: remove robot, table, background

    This class only include stage 2 and 3.
    '''

    def __init__(self, rm_pc_outliers_neighbors=25, selfgen=True):
        self.rm_pc_outliers_neighbors = 25
        self.rm_robot_type = 'box_keep_gripper'
        self.rotation_transform = RotationMatrixTransform()
        self.WORKSPACE = get_robot_workspace(real_robot=False, use_vlm=False)
        self.selfgen = selfgen

    def pcd_to_voxel(self,
                     xyz: np.ndarray,
                     rgb: np.ndarray,
                     voxel_size: float = 0.01  # 1cm
                     ) -> o3d.geometry.VoxelGrid:
        '''
        params:
            xyz: (N, 3),batch is fine
            rgb: (N, 3),batch is fine
        '''
        # some assert
        assert xyz.shape == rgb.shape
        # if batch
        b = None
        if len(xyz.shape) == 3:
            b = xyz.shape[0]
            xyz = xyz.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)

        # main
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        # main

        if b is not None:
            xyz = xyz.reshape(b, -1, 3)
            rgb = rgb.reshape(b, -1, 3)
        return voxel_grid

    def voxel_to_pcd(self,
                     voxel_grid: o3d.geometry.VoxelGrid,
                     ):
        pcd = voxel_grid.to_point_cloud()
        return pcd

    def pcd_voxel_pcd(self,
                      xyz: np.ndarray,
                      rgb: np.ndarray,
                      voxel_size: float = 0.01  # 1cm
                      ):
        voxel_grid = self.pcd_to_voxel(xyz, rgb, voxel_size)
        pcd = self.voxel_to_pcd(voxel_grid)
        return pcd

    def lotus_process(self):
        # remove robot
        # remove table
        # remove background

        pass

    def remove_table(self, xyz, rgb):
        TABLE_HEIGHT = 0.7505
        table_mask = xyz[:, 2] > TABLE_HEIGHT
        xyz = xyz[table_mask]
        rgb = rgb[table_mask]
        return xyz, rgb

    def remove_robot(self, xyz, rgb, arm_links_info, rm_robot_type='box_keep_gripper'):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False

        # main
        # get all points belongs to robot
        robot_box = RobotBox(arm_links_info, keep_gripper=keep_gripper, env_name='rlbench', selfgen=self.selfgen)

        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)

        robot_point_ids = np.array(list(robot_point_ids))

        mask = np.ones((xyz.shape[0], ), dtype=bool)

        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False

        # apply mask
        xyz = xyz[mask]
        rgb = rgb[mask]
        # /main

        return xyz, rgb

    def downsample(self, xyz, rgb, action_current, sample_points_by_distance, num_points=4096, downsample_type='random'):
        if len(xyz) > num_points:  # downsample
            if sample_points_by_distance:
                xyz, rgb = self._downsample_by_distance(xyz, rgb, action_current, num_points=num_points)
            else:
                xyz, rgb = self._downsample_random(xyz, rgb, action_current, num_points=num_points)
        else:
            # max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
            # point_idxs = np.random.permutation(len(xyz))[:max_npoints]
            # xyz = xyz[point_idxs]
            # rgb = rgb[point_idxs]
            pass
        return xyz, rgb

    def _dict_pos_probs(self,):
        pass

    def remove_outliers(self, xyz, rgb=None, return_idxs=False):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd, idxs = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd, idxs = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
        clf = LocalOutlierFactor(n_neighbors=self.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        if return_idxs:
            return xyz, rgb, idxs
        else:
            return xyz, rgb

    def _downsample_random(self, xyz, rgb, action_current, num_points=4096):
        point_idxs = np.random.choice(len(xyz), num_points, replace=False)
        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        return xyz, rgb

    def _downsample_by_distance(self, xyz, rgb, action_current, num_points=4096):
        dists = np.sqrt(np.sum((xyz - action_current[:3])**2, 1))
        probs = 1 / np.maximum(dists, 0.1)
        probs = np.maximum(softmax(probs), 1e-30)
        probs = probs / sum(probs)
        # probs = 1 / dists
        # probs = probs / np.sum(probs)
        point_idxs = np.random.choice(len(xyz), num_points, replace=False, p=probs)

        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        return xyz, rgb

    def augment_pc(self, flag, xyz, ee_pose, gt_action, gt_rot, aug_max_rot, rot_type, euler_resolution):
        if flag:
            angle = np.random.uniform(-1, 1) * aug_max_rot
            xyz = random_rotate_z(xyz, angle=angle)
            ee_pose[:3] = random_rotate_z(ee_pose[:3], angle=angle)
            gt_action[:3] = random_rotate_z(gt_action[:3], angle=angle)
            ee_pose[3:-1] = self._rotate_gripper(ee_pose[3:-1], angle)
            gt_action[3:-1] = self._rotate_gripper(gt_action[3:-1], angle)
            if rot_type == 'quat':
                gt_rot = gt_action[3:-1]
            elif rot_type == 'euler':
                gt_rot = self.rotation_transform.quaternion_to_euler(
                    torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy() / 180.
            elif rot_type == 'euler_disc':
                gt_rot = quaternion_to_discrete_euler(gt_action[3:-1], euler_resolution)
            elif rot_type == 'rot6d':
                gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                    torch.from_numpy(gt_action[3:-1][None, :]))[0].numpy()

            # add small noises (+-2mm)
            pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
            xyz = pc_noises + xyz

            return xyz, ee_pose, gt_action, gt_rot
        else:
            return xyz, ee_pose, gt_action, gt_rot

    def normalize_pc(self, xyz_shift, xyz_norm, xyz, action_current, action_next, gt_rot, height):
        if xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif xyz_shift == 'gripper':
            centroid = copy.deepcopy(action_current[:3])

        if xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        action_next[:3] = (action_next[:3] - centroid) / radius
        action_current[:3] = (action_current[:3] - centroid) / radius
        action_next = np.concatenate([action_next[:3], gt_rot, action_next[-1:]], 0)

        return centroid, radius, height, xyz, action_current, action_next

    def get_robot_point_idxs(self, pos_heatmap_no_robot, xyz, links_info):
        if pos_heatmap_no_robot:
            robot_box = RobotBox(
                arm_links_info=links_info,
                env_name='rlbench',
                selfgen=self.selfgen
            )
            robot_point_idxs = np.array(
                list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1])
            )
        else:
            robot_point_idxs = None
        return robot_point_idxs

    def get_pc_ft(self, xyz, rgb, height, use_height):
        rgb = rgb * 2 - 1
        pc_ft = np.concatenate([xyz, rgb], 1)
        if use_height:
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)
        return pc_ft

    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot

    def get_groundtruth_rotations(self, ee_poses, euler_resolution, rot_type='euler_disc'):
        gt_rots = torch.from_numpy(ee_poses.copy())   # quaternions
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

    def workspace(self, xyz):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        return in_mask


class ActionProcessorLotus:
    def __init__(self,):
        self.action = None

    pass
