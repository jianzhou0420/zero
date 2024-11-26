from act_jian.utils_all.utils_obs import normalize_position, normalize_image
from act_jian.utils_all.constants import JOINT_POSITIONS_LIMITS
from act_jian.utils_all.utils_depth import natural_sort_key
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import h5py
import clip
import re
# Utils


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


JOINT_POSITIONS_LIMITS = np.array([[-2.8973, 2.8973],
                                   [-1.7628, 1.7628],
                                   [-2.8973, 2.8973],
                                   [-3.0718, -0.0698],
                                   [-2.8973, 2.8973],
                                   [-0.0175, 3.7525],
                                   [-2.8973, 2.8973]])


def normalize_image(image):
    image = image / 255.0
    return image


def denormalize_image(image):
    image = image * 255.0
    return image


def normalize_position(position):
    if position.shape[0] == 8:
        position[:7] = (position[:7] - JOINT_POSITIONS_LIMITS[:, 0]) / (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0])
    else:
        position[..., :7] = (position[..., :7] - JOINT_POSITIONS_LIMITS[:, 0]) / (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0])

    return position


def denormalize_position(position):
    position[:7] = position[:7] * (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0]) + JOINT_POSITIONS_LIMITS[:, 0]
    return position


class JianRLBenchDataset(Dataset):
    '''
    load hdf5 files from the specified directory
    '''

    def __init__(self, action_chunk_size=200, data_dir=None):
        # TODO: config task
        self.action_chunk_size = action_chunk_size
        self.data_dir = data_dir
        self.episodes = sorted([d for d in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, d))], key=natural_sort_key)
        self.iteration_each_episode = np.zeros(len(self.episodes), dtype=int)

        joint_position_all = []
        for i in range(len(self.episodes)):
            with h5py.File(os.path.join(self.data_dir, f"episode_{i}.h5"), 'r') as f:
                self.iteration_each_episode[i] = f['joint_positions'].shape[0]
                joint_position_all.append(f['joint_positions'][:])
        self.accumulated_iteration = np.cumsum(self.iteration_each_episode)
        joint_position_all = np.concatenate(joint_position_all, axis=0)

        # check if all joint_position_all is within the limits
        assert np.all(joint_position_all >= JOINT_POSITIONS_LIMITS[:, 0]), "joint_position_limit error"
        assert np.all(joint_position_all <= JOINT_POSITIONS_LIMITS[:, 1]), "joint_position_limit error"

    def __len__(self):
        return self.accumulated_iteration[-1]

    def __getitem__(self, idx):
        return_dict = dict()  # a dict to give out 1. images, 2. current_position, 3. future_position, 4. is_data_mask
        # find the epoisode idx
        episode_idx = np.argmax(self.accumulated_iteration > idx)
        read_path = os.path.join(self.data_dir, f"episode_{episode_idx}.h5")
        frame_idx = idx - self.accumulated_iteration[episode_idx - 1] if episode_idx > 0 else idx

        # Firstly, retrieve data
        with h5py.File(read_path, 'r') as f:
            # get all images
            # need to firstly get some data to find the length of episode, despite unefficient
            actions_all_joints = f['joint_positions'][:]
            actions_all_gripper = f['gripper_open'][:].reshape(-1, 1)
            actions_all = np.concatenate((actions_all_joints, actions_all_gripper), axis=1)

            # sample one frame
            # first of all: image
            images_all = []
            images_all.append(f['left_shoulder_rgb'][frame_idx])
            images_all.append(f['right_shoulder_rgb'][frame_idx])
            images_all.append(f['wrist_rgb'][frame_idx])
            images_all.append(f['front_rgb'][frame_idx])
            images_all.append(f['overhead_rgb'][frame_idx])
            images_all = np.stack(images_all, axis=0)

            text_all = f['variation_descriptions '][:]

        # Secondly, prepare data
        # 先padding成action chunk的大小
        current_position = actions_all[frame_idx]
        future_position = actions_all[frame_idx + 1:]  # 8.17，排查错误。之前current_position=future_position[0],难道是这里导致的？
        length_future = len(future_position)

        # padding to action_chunk_size
        if length_future < self.action_chunk_size:
            future_position = np.concatenate((future_position, np.zeros((self.action_chunk_size - length_future, future_position.shape[1]))), axis=0, dtype=np.float32)
            is_pad = np.full(self.action_chunk_size, False)
            is_pad[length_future:] = True
        else:
            future_position = future_position[:self.action_chunk_size]
            is_pad = np.full(self.action_chunk_size, False)

        # norm with mean and std
        images_all = normalize_image(images_all)
        current_position = normalize_position(current_position)
        future_position = normalize_position(future_position)

        text = str(text_all[np.random.randint(len(text_all))])

        # concat images
        # images_all = torch.tensor(images_all).float()

        # return
        return_dict['images'] = torch.tensor(images_all).float()
        return_dict['current_position'] = torch.tensor(current_position).float()
        # return_dict['future_position'] = torch.tensor(future_position).float()
        # return_dict['is_pad'] = torch.tensor(is_pad).bool()
        return_dict['idx'] = torch.tensor(idx).int()
        return_dict['text'] = text

        return return_dict
