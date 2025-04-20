
import torch.nn.functional as F
from typing import Dict, Optional, Sequence
import json
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
import numpy as np
import pdb

import re


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

# Normalize joint position


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


# normalize gripper position


def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds


gripper_loc_bounds = torch.from_numpy(get_gripper_loc_bounds(
    "/media/jian/ssd4t/zero/assets/18_peract_tasks_location_bounds.json",
    buffer=0.04,
))


def normalize_pos(pos):
    pos_min = gripper_loc_bounds[0].float().to(pos.device)
    pos_max = gripper_loc_bounds[1].float().to(pos.device)
    return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0


def denormalize_pos(pos):
    pos_min = gripper_loc_bounds[0].float().to(pos.device)
    pos_max = gripper_loc_bounds[1].float().to(pos.device)
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


# rot
def normalise_quat(x: torch.Tensor):
    return x / torch.clamp(x.square().sum(dim=-1).sqrt().unsqueeze(-1), min=1e-10)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_ortho6d_from_rotation_matrix(matrix):
    # The orhto6d represents the first two column vectors a1 and a2 of the
    # rotation matrix: [ | , |,  | ]
    #                  [ a1, a2, a3]
    #                  [ | , |,  | ]
    ortho6d = matrix[:, :, :2].permute(0, 2, 1).flatten(-2)
    return ortho6d


def convert_rot(signal):
    rotation_parametrization = '6D'
    quaternion_format = 'wxyz'
    signal[..., 3:7] = normalise_quat(signal[..., 3:7])
    if rotation_parametrization == '6D':
        # The following code expects wxyz quaternion format!
        if quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
        rot = quaternion_to_matrix(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot_6d = get_ortho6d_from_rotation_matrix(rot)
            rot_6d = rot_6d.reshape(B, L, 6)
        else:
            rot_6d = get_ortho6d_from_rotation_matrix(rot)
        signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
    return signal


def normalize_vector(v, return_mag=False):
    device = v.device
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out  # batch*3


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def deconvert_rot(signal):
    rotation_parametrization = '6D'
    quaternion_format = 'wxyz'
    if rotation_parametrization == '6D':
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        # The above code handled wxyz quaternion format!
        if quaternion_format == 'xyzw':
            signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
    return signal
