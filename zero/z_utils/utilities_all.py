
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
import numpy as np


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
    for feat in features:
        current_length, dim = feat.shape
        # Create a new array of zeros with target_length rows and same feature dimension.
        padded = np.zeros((target_length, dim), dtype=feat.dtype)
        # Fill the first current_length rows with the feature data.
        padded[:current_length, :] = feat
        padded_features.append(padded)

    # Optionally stack into one numpy array (batch_size, target_length, 512)
    return np.stack(padded_features)


# region Joint Position

JOINT_POSITION_LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
                         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1]]


def normalize_theta_positions(theta_positions):
    lower_limit = torch.tensor(JOINT_POSITION_LIMITS[0])
    upper_limit = torch.tensor(JOINT_POSITION_LIMITS[1])
    return (theta_positions - lower_limit) / (upper_limit - lower_limit)


def denormalize_theta_positions(normalized_theta_positions):
    lower_limit = JOINT_POSITION_LIMITS[0, :]
    upper_limit = JOINT_POSITION_LIMITS[1, :]
    return normalized_theta_positions * (upper_limit - lower_limit) + lower_limit

# endregion
