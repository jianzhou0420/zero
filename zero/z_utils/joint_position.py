import numpy as np
import torch

JOINT_POSITION_LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
                         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1]]


def normalize_theta_positions(theta_positions):
    if type(theta_positions) == np.ndarray:
        shape = theta_positions.size
    elif type(theta_positions) == torch.Tensor:
        shape = theta_positions.size()
    else:
        raise ValueError
    if len(shape) == 2:
        theta_positions = theta_positions.unsqueeze(0)

    lower_limit = JOINT_POSITION_LIMITS[0]
    upper_limit = JOINT_POSITION_LIMITS[1]
    return (theta_positions - lower_limit) / (upper_limit - lower_limit)


def denormalize_theta_positions(normalized_theta_positions):
    lower_limit = JOINT_POSITION_LIMITS[0, :]
    upper_limit = JOINT_POSITION_LIMITS[1, :]
    return normalized_theta_positions * (upper_limit - lower_limit) + lower_limit
