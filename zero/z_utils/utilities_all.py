
from typing import Callable
import open3d as o3d
import einops
import torch.nn.functional as F
from typing import Dict, Optional, Sequence
import json
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
import numpy as np
import pdb

import re

# tools


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


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


# type
def tensorfp32(x):
    if torch.is_tensor(x):
        x = x.float()
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif isinstance(x, list):
        x = torch.from_numpy(np.array(x)).float()
    return x


def npafp32(x):
    return np.array(x, dtype=np.float32)

# other


class Resize:
    """Resize and pad/crop the image and aligned point cloud."""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs):
        """Accept tensors as T, N, C, H, W."""
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Sample resize scale from continuous range
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
            )
            for n, arg in kwargs.items()
        }

        # If resized image is smaller than original, pad it with a reflection
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_w - resized_size[1], 0), max(
                raw_h - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # If resized image is larger than original, crop it
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )
        kwargs = {
            n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()
        }

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs


def get_robot_pcd_idx(xyz, obbox):
    points = o3d.utility.Vector3dVector(xyz)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = points
    # o3d.visualization.draw_geometries([pcd, *obbox])
    robot_point_idx = set()
    for box in obbox:
        tmp = box.get_point_indices_within_bounding_box(points)
        robot_point_idx = robot_point_idx.union(set(tmp))
    robot_point_idx = np.array(list(robot_point_idx))
    mask = np.zeros(len(xyz), dtype=bool)
    mask[robot_point_idx] = True
    return mask


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
