import numpy as np
from scipy.spatial.transform import Rotation as R
from codebase.z_utils.Rotation import *


cameraPose = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]


Euler_rot_change = np.array([
    [0, 0, -30],  # 右转
    [0, 0, -60],
    [0, 0, -270],
    [0, 30, 0],  # 低头
    [0, -30, 0],  # 抬头
])


def rotate_camera_world_frame(cameraPose, Euler_rot_change):
    """
    Rotate the camera in the world frame.
    :param cameraPose: [x, y, z, qx, qy, qz, qw]
    :param Euler_rot_change: [roll, pitch, yaw]
    :return: new camera pose
    """
    # Convert quaternion to rotation matrix
    R_camera_old = quat2mat(cameraPose[3:])

    # Convert Euler angles to rotation matrix
    Matrix_rot_change = euler2mat(np.radians(Euler_rot_change))

    # Calculate new rotation matrix
    R_camera_new = Matrix_rot_change @ R_camera_old

    # Convert rotation matrix back to quaternion
    new_quat = mat2quat(R_camera_new)

    # Return new camera pose
    return np.concatenate((cameraPose[:3], new_quat))


def rotate_camera_body_frame(cameraPose, Euler_rot_change):
    """
    Rotate the camera in the body frame.
    :param cameraPose: [x, y, z, qx, qy, qz, qw]
    :param Euler_rot_change: [roll, pitch, yaw]
    :return: new camera pose
    """
    # Convert quaternion to rotation matrix
    R_camera_old = quat2mat(cameraPose[3:])

    # Convert Euler angles to rotation matrix
    Matrix_rot_change = euler2mat(np.radians(Euler_rot_change))

    # Calculate new rotation matrix
    R_camera_new = R_camera_old @ Matrix_rot_change

    # Convert rotation matrix back to quaternion
    new_quat = mat2quat(R_camera_new)

    # Return new camera pose
    return np.concatenate((cameraPose[:3], new_quat))
