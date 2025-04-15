
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


JOINT_POSITION_LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0],
                         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1]]


def normalize_JP(JP):
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
