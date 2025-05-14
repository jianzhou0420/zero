import numpy as np


def mask2idx(mask):
    """
    Convert a boolean mask to indices.

    Args:
        mask (np.ndarray): A boolean array.

    Returns:
        np.ndarray: An array of indices where the mask is True.
    """
    return np.where(mask)[0]


def idx2mask(idx, length):
    """
    Convert indices to a boolean mask.

    Args:
        idx (np.ndarray): An array of indices.
        length (int): The length of the mask.

    Returns:
        np.ndarray: A boolean array where the indices are True.
    """
    if not isinstance(idx, np.ndarray):
        idx = np.array(idx)
    mask = np.zeros(length, dtype=bool)
    mask[idx] = True
    return mask
