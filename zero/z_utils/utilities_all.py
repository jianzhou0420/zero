
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
import numpy as np
import pdb


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def summary_models(model, return_flag=False):
    summary = ModelSummary(model)
    if return_flag:
        return summary
    else:
        print(summary)


def pad_clip_features(features, target_length=77):
    """
    Pads a list of CLIP feature arrays (each of shape (L, 512)) to a fixed target length.

    Args:
        features (list of np.array): List of feature arrays with shape (L, 512), where L can vary.
        target_length (int): The target sequence length (default is 77).

    Returns:
        np.array: A numpy array of shape (batch_size, target_length, 512) where each feature array
                  has been padded with zeros if necessary.
    """
    padded_features = []
    mask = []
    for feat in features:
        current_length, dim = feat.shape

        padded = np.zeros((target_length, dim), dtype=feat.dtype)
        mask_s = np.zeros((target_length,), dtype=bool)

        padded[:current_length, :] = feat
        mask_s[:current_length] = True

        padded_features.append(padded)
        mask.append(mask_s)

    # Stack the padded features into a single numpy array.
    padded_features = np.stack(padded_features, axis=0)
    mask = np.stack(mask, axis=0, dtype=bool)
    return padded_features, mask


# region Joint Position

JOINT_POSITION_LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
                         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1]]


def normalize_theta_positions(theta_positions):
    # must be batch

    lower_limit = torch.tensor(JOINT_POSITION_LIMITS[0], device=theta_positions.device).repeat(theta_positions.shape[0], 1)
    upper_limit = torch.tensor(JOINT_POSITION_LIMITS[1], device=theta_positions.device).repeat(theta_positions.shape[0], 1)

    # print(lower_limit.shape)
    # print(upper_limit.shape)
    # print(theta_positions.shape)
    # print('theta_position', theta_positions[0, :])
    test1 = theta_positions - lower_limit
    test2 = upper_limit - lower_limit
    test3 = (test1 / test2) * 2 - 1
    # print(test3.shape)
    # print('afterprocess', test3[0, :])
    return test3


def denormalize_theta_positions(normalized_theta_positions):
    lower_limit = JOINT_POSITION_LIMITS[0, :]
    upper_limit = JOINT_POSITION_LIMITS[1, :]

    test = (normalized_theta_positions + 1) / 2 * (upper_limit - lower_limit) + lower_limit
    return test

# endregion


def normalize_theta_positions(theta_positions):
    # must be batch

    lower_limit = torch.tensor(JOINT_POSITION_LIMITS[0], device=theta_positions.device).repeat(theta_positions.shape[0], 1)
    upper_limit = torch.tensor(JOINT_POSITION_LIMITS[1], device=theta_positions.device).repeat(theta_positions.shape[0], 1)

    # print(lower_limit.shape)
    # print(upper_limit.shape)
    # print(theta_positions.shape)
    # print('theta_position', theta_positions[0, :])
    test1 = theta_positions - lower_limit
    test2 = upper_limit - lower_limit
    test3 = test1 / test2
    # print(test3.shape)
    # print('afterprocess', test3[0, :])
    return test3


def denormalize_theta_positions(normalized_theta_positions):
    lower_limit = JOINT_POSITION_LIMITS[0, :]
    upper_limit = JOINT_POSITION_LIMITS[1, :]

    test = normalized_theta_positions * (upper_limit - lower_limit) + lower_limit
    return test
