from zero.v2.models.lotus.utils.robot_box import RobotBox
from rlbench.demo import Demo
from typing import List, Dict, Optional, Sequence, Tuple, TypedDict, Union, Any
from typing import List, Tuple

import numpy as np

import torch
import einops
import json
from scipy.spatial.transform import Rotation as R
import re


def convert_gripper_pose_world_to_image(obs, camera: str) -> Tuple[int, int]:
    '''Convert the gripper pose from world coordinate system to image coordinate system.
    image[v, u] is the gripper location.
    '''
    extrinsics_44 = obs.misc[f"{camera}_camera_extrinsics"].astype(np.float32)
    extrinsics_44 = np.linalg.inv(extrinsics_44)

    intrinsics_33 = obs.misc[f"{camera}_camera_intrinsics"].astype(np.float32)
    intrinsics_34 = np.concatenate([intrinsics_33, np.zeros((3, 1), dtype=np.float32)], 1)

    gripper_pos_31 = obs.gripper_pose[:3].astype(np.float32)[:, None]
    gripper_pos_41 = np.concatenate([gripper_pos_31, np.ones((1, 1), dtype=np.float32)], 0)

    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31[:, 0]

    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


def quaternion_to_discrete_euler(quaternion, resolution: int):
    euler = R.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution: int):
    euluer = (discrete_euler * resolution) - 180
    return R.from_euler('xyz', euluer, degrees=True).as_quat()


def euler_to_quat(euler, degrees):
    rotation = R.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = R.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)


'''Identify way-point in each RLBench Demo
'''


def _is_stopped(demo, i, obs, stopped_buffer):
    next_is_not_final = (i < (len(demo) - 2))
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and next_is_not_final
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def quaternion_to_discrete_euler(quaternion, resolution: int):
    euler = R.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution: int):
    euluer = (discrete_euler * resolution) - 180
    return R.from_euler('xyz', euluer, degrees=True).as_quat()


def euler_to_quat(euler, degrees):
    rotation = R.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = R.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)


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
