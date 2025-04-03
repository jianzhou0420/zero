import numpy as np
import torch

JOINT_POSITION_LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
                         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1]]


def normaliza_JP(JP):
    lower = JOINT_POSITION_LIMITS[0]
    upper = JOINT_POSITION_LIMITS[1]
    if isinstance(JP, np.ndarray):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        # broadcasting
        normalized_JP = 2 * (JP - lower_np) / (upper_np - lower_np) - 1

    elif isinstance(JP, torch.Tensor):
        lower_tensor = torch.tensor(lower, device=JP.device)
        upper_tensor = torch.tensor(upper, device=JP.device)
        # broadcasting
        normalized_JP = 2 * (JP - lower_tensor) / (upper_tensor - lower_tensor) - 1
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return normalized_JP


def denormalize_JP(norm_JP):
    lower = JOINT_POSITION_LIMITS[0]
    upper = JOINT_POSITION_LIMITS[1]
    if isinstance(norm_JP, np.ndarray):
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        # broadcasting
        JP = lower_np + (norm_JP + 1) / 2 * (upper_np - lower_np)

    elif isinstance(norm_JP, torch.Tensor):
        lower_t = torch.tensor(lower, device=norm_JP.device)
        upper_t = torch.tensor(upper, device=norm_JP.device)
        # broadcasting
        JP = lower_t + (norm_JP + 1) / 2 * (upper_t - lower_t)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return JP
