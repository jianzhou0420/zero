import open3d as o3d
import pickle
import re
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from zero.expForwardKinematics.ObsProcessor.ObsProcessorPtv3_fk import ObsProcessorPtv3
import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import einops

from copy import deepcopy as copy
import json
import random
from zero.expForwardKinematics.ReconLoss.ForwardKinematics import FrankaEmikaPanda

# --------------------------------------------------------------
# region Dataset


class DatasetFK(Dataset):
    def __init__(self, config, data_dir=None):
        super().__init__()
        '''
        如果没有cache_dataset_init_path,那么就从头开始
        
        '''
        self.config = config
        data_dir = data_dir  # 因为namesapce不高亮，所以尽量用字典的方式，方便区分

        if config['TrainDataset']['cache_dataset_init_path'] is not None:
            cache_dataset_init_path = config['TrainDataset']['cache_dataset_init_path']
            with open(cache_dataset_init_path, 'rb') as f:
                cache_init = pickle.load(f)

            (self.g_episode_to_taskvar,
             self.g_episode_to_path,
             self.g_episode_to_l_episode,
             self.frames,
             self.g_frame_to_taskvar,
             self.g_frame_to_g_episode,
             self.g_frame_to_frame,
             self.g_frame_to_l_episode) = cache_init

        else:
            cache_init = None

        if cache_init is None:
            # tasks_to_use
            tasks_to_use = config['TrainDataset']['tasks_to_use']
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
                    variation_folder_path = os.path.join(task_folder_path, variation_folder)
                    episodes_list = sorted(os.listdir(variation_folder_path), key=natural_sort_key)
                    for episode_folder in episodes_list:
                        episode_folder_path = os.path.join(variation_folder_path, episode_folder)
                        self.g_episode_to_path.append(episode_folder_path)
                        variation_id = int(variation_folder.split('variation')[-1])
                        taskvar = task_name + '_peract' + '+' + str(variation_id)
                        self.g_episode_to_taskvar.append(taskvar)
                        with open(os.path.join(episode_folder_path, 'data.pkl'), 'rb') as f:
                            data = pickle.load(f)
                        self.frames.append(len(data['rgb']))
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

            cache_init = (self.g_episode_to_taskvar,
                          self.g_episode_to_path,
                          self.g_episode_to_l_episode,
                          self.frames,
                          self.g_frame_to_taskvar,
                          self.g_frame_to_g_episode,
                          self.g_frame_to_frame,
                          self.g_frame_to_l_episode)

            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_root = '/data/zero/1_Data/D_Cache_init'
            save_path = os.path.join(save_root, current_time + 'cache_dataset_init_path.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(cache_init, f)
            print('cache_dataset_init_path.pkl has been saved')

        # 3.container
        self.cache = dict()
        self.max_cache_length = 300
        print(f"max_cache_length: {self.max_cache_length}")
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()

        # 4.franka
        # self.franka = FrankaEmikaPanda()
        # 5. obs_processor
        self.obs_processor = ObsProcessorPtv3(config, train_flag=True)
        self.obs_processor._dataset_init_FK()

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
        '''
            data={
                'xyz': (N, 3),
                'rgb': (N, 3),
                'JP_hist': (N, 7),
                'JP_futr': (N, 7),
            }
        '''
        outs = {
            'pc_fts': [],
            'JP_hist': [],
            'JP_futr': [],
            'instr': [],
            'instr_mask': [],
            'noncollision_mask': [],
        }
        if self.config['test']:
            g_episode = 0
        data = self.check_cache(g_episode)
        outs = self.obs_processor.dynamic_process_fk(data, taskvar=self.g_episode_to_taskvar[g_episode])
        return outs
    # endregion


def ptv3_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])

    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)  # (#all points, 6)

    for key in ['ee_poses', 'gt_actions']:
        batch[key] = torch.stack(batch[key], 0)

    # if 'disc_pos_probs' in batch:
    #     batch['disc_pos_probs'] = batch['disc_pos_probs'] # [(3, #all pointspos_bins*2)]

    batch['step_ids'] = torch.LongTensor(batch['step_ids'])

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    return batch


def collect_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)  # (#all points, 6)

    for key in ['JP_hist', 'JP_futr', 'instr', 'instr_mask']:
        batch[key] = torch.stack(batch[key], 0)

    return batch
# --------------------------------------------------------------
# region utils


def tensorfp32(x):
    x = torch.tensor(x, dtype=torch.float32)
    return x


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


if __name__ == '__main__':
    from zero.expForwardKinematics.config.default import get_config
    from torch.utils.data import DataLoader, Dataset
    config_path = '/data/zero/zero/expForwardKinematics/config/FK.yaml'
    config = get_config(config_path)
    data_dir = '/data/zero/1_Data/B_Preprocess/FK/1000_train_eval/train'
    dataset = DatasetFK(config, data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collect_fn)
    data1 = next(iter(loader))
