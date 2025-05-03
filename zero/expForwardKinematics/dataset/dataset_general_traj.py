from torch.utils.data import Dataset, DataLoader, Sampler
import open3d as o3d
import pickle
import re
from torch.utils.data import Dataset
import torch
import os
import numpy as np

import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import einops
import copy
import json

import sys
import random
from zero.expForwardKinematics.ReconLoss.ForwardKinematics import FrankaEmikaPanda
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import natural_sort_key
# --------------------------------------------------------------
# region Dataset


class DatasetGeneral_traj(Dataset):
    def __init__(self, config, data_dir: str, ObsProcessor: ObsProcessorRLBenchBase):
        super().__init__()
        '''
        如果没有cache_dataset_init_path,那么就从头开始
        '''
        self.config = config
        data_dir = data_dir  # 因为namesapce不高亮，所以尽量用字典的方式，方便区分

        # tasks_to_use
        tasks_to_use = config['TrainDataset']['tasks_to_use']
        tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
        tasks_all = [t for t in tasks_all if t in tasks_to_use] if tasks_to_use is not None else tasks_all
        # 1. episodes-wise list
        self.g_episode_to_path = []  # retrieve all episodes path and put them in self.episodes
        self.frames = []  # How many frames in each episode
        for task_name in tasks_all:
            task_folder_path = os.path.join(data_dir, task_name)
            variation_list = sorted(os.listdir(task_folder_path), key=natural_sort_key)
            for variation_folder in variation_list:
                variation_folder_path = os.path.join(task_folder_path, variation_folder)
                if len(os.listdir(variation_folder_path)) <= 1:
                    variation_folder_path = os.path.join(task_folder_path, variation_folder, 'episodes')
                episodes_list = sorted(os.listdir(variation_folder_path), key=natural_sort_key)
                for episode_folder in episodes_list:
                    episode_folder_path = os.path.join(variation_folder_path, episode_folder)
                    self.g_episode_to_path.append(episode_folder_path)

        # 2. data allocation
        self.inner_counter = 0
        self.num_episodes = len(self.g_episode_to_path)
        self.num_workers = 4
        self.num_max_frames = 200
        self.num_iter = self.num_episodes / self.num_workers

        self.num_epoches = config['Trainer']['epoches']

        self.episode_map = [
            [] for _ in range(self.num_workers)
        ]

        # allocate the episodes to each worker
        for i in range(self.num_epoches):
            candidate_episodes = self.g_episode_to_path.copy()
            random.shuffle(candidate_episodes)
            base_count = len(candidate_episodes) // self.num_workers
            remainder = len(candidate_episodes) % self.num_workers
            idx = 0
            for j in range(self.num_workers):
                count = base_count + (1 if j < remainder else 0)  # 前remainder个人多分一个
                self.episode_map[j].append(candidate_episodes[idx:idx + count])
                idx += count

        self.counter = 0
        self.cache = dict()
        self.max_cache_length = 2

        # 3. obs processor
        self.obs_processor = ObsProcessor(config, train_flag=True)  # type: ObsProcessorRLBenchBase
        self.obs_processor.dataset_init()

    def check_cache_with_path(self, g_episode_path):
        episode_name = g_episode_path.split('/')[-1]
        if self.cache.get(episode_name) is None:
            print(f'loading data from disk{episode_name}')
            with open(os.path.join(g_episode_path, 'data.pkl'), 'rb') as f:
                data = pickle.load(f)
            if len(self.cache) >= self.max_cache_length:
                first_key = next(iter(self.cache))
                self.cache.pop(first_key)
            self.cache[episode_name] = data
            return data
        else:
            # print('using cache')
            return self.cache[episode_name]

    def __len__(self):
        return int(self.num_iter * self.num_workers * self.num_max_frames)

    def __getitem__(self, idx):

        # Step 1: get data from cache
        # identify the worker, epoch, and iteration
        iter_idx = idx // (self.num_workers * self.num_max_frames)
        # worker_idx_infer = idx % (self.num_workers * self.num_max_frames) % self.num_workers
        worker_idx_gt = torch.utils.data.get_worker_info().id
        epoch_idx = int(self.counter // (np.ceil(self.num_episodes / self.num_workers) * self.num_max_frames))
        self.counter += 1

        try:
            episode_path = self.episode_map[worker_idx_gt][epoch_idx][iter_idx]
            pass
        except IndexError:  # 每个worker分到的episode数量不一样，正常现象。
            return None
        data = self.check_cache_with_path(episode_path)
        out = self.obs_processor.dynamic_process(data)
        return out

    # endregion


# class Sampler_Traj(Sampler):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset

#     def __iter__(self):
#         # 返回一个迭代器，指定按哪些索引顺序加载数据
#         return iter(self.indices)

#     def __len__(self):
#         return len(self.indices)


def worker_init_fn(worker_id):
    pass

    # 输出当前 worker 处理的样本范围


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# endregion
if __name__ == '__main__':
    from zero.expForwardKinematics.config.default import get_config
    from torch.utils.data import DataLoader, Dataset
    from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP import ObsProcessorDP
    from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP_traj import ObsProcessorDP_traj

    config_path = '/media/jian/ssd4t/zero/zero/expForwardKinematics/config/DP_0501_01.yaml'
    config = get_config(config_path)
    data_dir = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DP_traj/trajectory/test2/42'
    dataset = DatasetGeneral_traj(config, data_dir, ObsProcessor=ObsProcessorDP_traj)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,)
    for i in range(len(loader)):
        data1 = next(iter(loader))
        if i == 20:
            break
