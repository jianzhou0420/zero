import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import h5py
import clip
import re
# Utils
import pickle


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

    def __init__(self, data_dir=None):

        self.data_dir = os.path.join(data_dir, 'train')
        shape_list_dir = os.path.join(data_dir, 'dataset_shape_list.pkl')
        self.dataset_shape_list = pickle.load(open(shape_list_dir, 'rb'))

    def __len__(self):
        return len(list(d for d in os.listdir(self.data_dir) if d.endswith('.pkl')))

    def __getitem__(self, idx):
        idx_tmp = idx
        task_idx, episode_idx, frame_idx = self._get_idx(self.dataset_shape_list, idx_tmp)

        return_dict = dict()  # a dict to give out 1. images, 2. current_position, 3. future_position, 4. is_data_mask
        # find the epoisode idx
        data_dict = pickle.load(open(os.path.join(self.data_dir, f'{idx_tmp}.pkl'), 'rb'))

        # get all images
        # need to firstly get some data to find the length of episode, despite unefficient

        actions_all_joints = data_dict['joint_positions'][:]
        actions_all_gripper = np.array(data_dict['gripper_open'][:]).reshape(-1, 1)
        actions_all = np.concatenate((actions_all_joints, actions_all_gripper), axis=1)

        # sample one frame
        # first of all: image
        images_all = []
        images_all.append(data_dict['left_shoulder_rgb'])
        images_all.append(data_dict['right_shoulder_rgb'])
        images_all.append(data_dict['wrist_rgb'])
        images_all.append(data_dict['front_rgb'])
        images_all.append(data_dict['overhead_rgb'])
        images_all = np.stack(images_all, axis=0)

        text_all = data_dict['variation_descriptions'][:]

        # Secondly, prepare data
        # 先padding成action chunk的大小
        current_position = actions_all[frame_idx].reshape(-1, 8)
        future_position = actions_all[frame_idx + 1:].reshape(-1, 8)  # single frame

        if future_position.shape[0] == 0:
            future_position = current_position
        else:
            future_position = future_position[0].reshape(-1, 8)
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
        return_dict['future_position'] = torch.tensor(future_position).float()
        # return_dict['is_pad'] = torch.tensor(is_pad).bool()
        return_dict['idx'] = torch.tensor(frame_idx).int()
        return_dict['text'] = text

        return return_dict

    def _get_idx(self, dataset_shape_list, idx):
        idx_tmp = idx
        for i, this_list in enumerate(dataset_shape_list):
            # print('this_list=', this_list)
            for j, frames_num in enumerate(this_list):
                idx_tmp = idx_tmp - frames_num
                if idx_tmp < 0:
                    task_idx = i
                    episode_idx = j
                    frame_idx = frames_num + idx_tmp
                    return task_idx, episode_idx, frame_idx


if __name__ == "__main__":

    with open('/data/rlbench_original/train/close_jar/all_variations/episodes/episode0/low_dim_obs.pkl', 'rb') as f:
        test = pickle.load(f)
    print(test)
    pass
