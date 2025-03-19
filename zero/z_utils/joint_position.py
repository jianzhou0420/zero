import numpy as np

JOINT_POSITIONS_LIMITS = np.array([[-2.8973, 2.8973],
                                   [-1.7628, 1.7628],
                                   [-2.8973, 2.8973],
                                   [-3.0718, -0.0698],
                                   [-2.8973, 2.8973],
                                   [-0.0175, 3.7525],
                                   [-2.8973, 2.8973],
                                   [0, 1]]).T  # gripper


def normalize_theta_positions(theta_positions):
    lower_limit = JOINT_POSITIONS_LIMITS[0, :]
    upper_limit = JOINT_POSITIONS_LIMITS[1, :]
    return (theta_positions - lower_limit) / (upper_limit - lower_limit)


def denormalize_theta_positions(normalized_theta_positions):
    lower_limit = JOINT_POSITIONS_LIMITS[0, :]
    upper_limit = JOINT_POSITIONS_LIMITS[1, :]
    return normalized_theta_positions * (upper_limit - lower_limit) + lower_limit
