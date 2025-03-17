
import pickle
import re
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from zero.expAugmentation.ObsProcessor.ObsProcessorPtv3 import ObsProcessorPtv3
# --------------------------------------------------------------
# region tools


def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))


def pad_tensors(tensors, lens=None, pad=0, max_len=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens) if max_len is None else max_len
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or nparray int, shape=(N, )
    Returns:
        masks: nparray, shape=(N, L), padded=0
    """
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# endregion

# --------------------------------------------------------------
# region dataset


class Dataset_DP_PTV3(Dataset):
    def __init__(self, config, data_dir=None):
        super().__init__()
        self.config = config

        self.obs_processor = ObsProcessorPtv3(config)
        self.obs_processor._dataset_init()

        data_dir = data_dir  # 因为namesapce不高亮，所以尽量用字典的方式，方便区分
        tasks_to_use = config['TRAIN_DATASET']['tasks_to_use']

        tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
        tasks_all = [t for t in tasks_all if t in tasks_to_use] if tasks_to_use is not None else tasks_all

        # 1. episodes-wise list
        self.g_episode_to_taskvar = []  # Which taskvar is each episode
        self.g_episode_to_path = []  # retrieve all episodes path and put them in self.episodes
        self.g_episode_to_l_episode = []  # Which episode in each taskvar
        self.frames = []  # How many frames in each episode
        for task_name in tasks_all:
            task_folder_path = os.path.join(data_dir, task_name)
            variation_list = sorted(os.listdir(task_folder_path), key=natural_sort_key)
            for variation_folder in variation_list:
                l_episode = 0
                variation_folder_path = os.path.join(task_folder_path, variation_folder, 'episodes')
                episodes_list = sorted(os.listdir(variation_folder_path), key=natural_sort_key)
                for episode_folder in episodes_list:
                    episode_folder_path = os.path.join(variation_folder_path, episode_folder)
                    self.g_episode_to_path.append(episode_folder_path)
                    variation_id = int(variation_folder.split('variation')[-1])
                    taskvar = task_name + '_peract' + '+' + str(variation_id)
                    self.g_episode_to_taskvar.append(taskvar)
                    with open(os.path.join(episode_folder_path, 'data.pkl'), 'rb') as f:
                        data = pickle.load(f)
                    self.frames.append(len(data['data_ids']))

                    self.g_episode_to_l_episode.append(l_episode)
                    l_episode += 1
        # 2. frame-wise list
        self.g_frame_to_taskvar = []
        self.g_frame_to_g_episode = []
        self.g_frame_to_frame = []
        self.g_frame_to_l_episode = []

        for episode_id, frame in enumerate(self.frames):
            self.g_frame_to_g_episode.extend([episode_id] * frame)
            self.g_frame_to_taskvar.extend([self.g_episode_to_taskvar[episode_id]] * frame)
            self.g_frame_to_frame.extend(list(range(frame)))
            self.g_frame_to_l_episode.extend([episode_id] * frame)

        # 3.container
        self.cache = dict()
        self.max_cache_length = 1800
        print(f"max_cache_length: {self.max_cache_length}")
        for i in range(len(self.g_episode_to_path)):
            self.check_cache(i)

    def check_cache(self, g_episode):
        if self.cache.get(g_episode) is None:
            episode_path = self.g_episode_to_path[g_episode]
            with open(os.path.join(episode_path, 'data.pkl'), 'rb') as f:
                data = pickle.load(f)
            if len(self.cache) >= self.max_cache_length:
                first_key = next(iter(self.cache))
                self.cache.pop(first_key)
            self.cache[g_episode] = data
            return data
        else:
            return self.cache[g_episode]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, g_episode):
        taskvar = self.g_episode_to_taskvar[g_episode]
        obs_static = self.check_cache(g_episode)
        obs_dynamic = self.obs_processor.dynamic_process(obs_static, taskvar)
        return obs_dynamic
        # Load data

# endregion


if __name__ == '__main__':
    from zero.expAugmentation.config.default import get_config
    confi_path = '/media/jian/ssd4t/zero/zero/expAugmentation/config/DP.yaml'
    config = get_config(confi_path)
    data_dir = os.path.join(config['TRAIN_DATASET']['data_dir'], 'train')
    dataset = Dataset_DP_PTV3(config, data_dir)
    print(len(dataset))
    print(dataset[0].keys())
