from scipy.spatial.transform import Rotation as R
import numpy as np

# Example
# def euler_to_quaternion(euler):
#     r = R.from_euler('xyz', euler, degrees=True)
#     return r.as_quat()


# def quaternion_to_euler(quat):
#     r = R.from_quat(quat)
#     return r.as_euler('xyz', degrees=True)


# def quaternion_to_rotation_matrix(quat):
#     r = R.from_quat(quat)
#     return r.as_matrix()
# /Example


def obs_gripper_pose_to_homo_matrix(obs):
    translation = obs.gripper_pose[:3]
    quaternion = obs.gripper_pose[3:]
    rotation = R.from_quat(quaternion).as_matrix()
    homogenous_matrix = np.eye(4)
    homogenous_matrix[:3, :3] = rotation
    homogenous_matrix[:3, 3] = translation
    return homogenous_matrix


'''
just examples, use R directly instead of this function
'''
