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
import random
from zero.expForwardKinematics.ReconLoss.ForwardKinematics import FrankaEmikaPanda
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import natural_sort_key

# --------------------------------------------------------------
# region Dataset


class DatasetGeneral(Dataset):
    def __init__(self, config, data_dir: str, ObsProcessor: ObsProcessorRLBenchBase):
        super().__init__()
        '''
        如果没有cache_dataset_init_path,那么就从头开始
        '''

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
        data = self.check_cache(g_episode)
        outs = self.obs_processor.dynamic_process(data, taskvar=self.g_episode_to_taskvar[g_episode])
        return outs
    # endregion


if __name__ == '__main__':
    from zero.expForwardKinematics.config.default import get_config
    from torch.utils.data import DataLoader, Dataset
    from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP import ObsProcessorDP
    config_path = '/media/jian/ssd4t/zero/zero/expForwardKinematics/config/DP_0501_01.yaml'
    config = get_config(config_path)
    data_dir = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DP/keypose/singleVar/train'
    dataset = DatasetGeneral(config, data_dir, ObsProcessorDP)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.obs_processor.collect_fn)
    data1 = next(iter(loader))
