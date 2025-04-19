
import pickle
import re
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from zero.expForwardKinematics.ObsProcessor.ObsProcessorDA3D import ObsProcessorDA3D
import time
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import einops
from zero.z_utils.utilities_all import pad_clip_features, normalize_JP
import copy

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


# endregion

# --------------------------------------------------------------
# region Dataset


class DatasetDA3D(Dataset):
    def __init__(self, config, data_dir=None):
        super().__init__()
        '''
        如果没有cache_dataset_init_path,那么就从头开始
        
        '''
        self.config = config
        data_dir = data_dir  # 因为namesapce不高亮，所以尽量用字典的方式，方便区分

        self.obs_processor = ObsProcessorDA3D(config, data_dir)
        self.obs_processor._dataset_init_DA3D()

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
        self.max_cache_length = 50
        print(f"max_cache_length: {self.max_cache_length}")
        # for i in range(len(self.g_episode_to_path)):
        #     if len(self.cache) >= self.max_cache_length:
        #         break
        #     self.check_cache(i)

        # 4, other
        self._resize = Resize(config['TrainDataset']['image_rescales'])

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
            batch:{
                'rgb': torch.Tensor (B, ncam,3, H, W)
                'xyz': torch.Tensor (B, ncam,3, H, W)
                'action_history': torch.Tensor (B, history, naction)
                'action_future': torch.Tensor (B, horizon, naction)
                'joint_position_future': torch.Tensor (B, horizon, njoint+open) 第一帧是current position,最后一帧是下一个keypose的position
                'joint_position_history': torch.Tensor (B, history, njoint+open) 最后一帧是current position
                'timestep': torch.Tensor (B,)
                'instruction': torch.Tensor (B, max_instruction_length, dim)
            }

        '''
        outs = {
            'rgb': None,
            'pcd': None,
            'joint_position_history': None,
            'joint_position_future': None,
            'instruction': None
        }

        data = self.check_cache(g_episode)
        taskvar = self.g_frame_to_taskvar[g_episode]

        outs = self.obs_processor.dynamic_process_DA3D(data, taskvar)

        # 暂时只要了 rgb,pcd,joint_position_history,joint_position_future和txt

        return outs
# endregion


def collect_fn(batch):
    collated = {}
    for key in batch[0]:
        # Concatenate the tensors from each dict in the batch along dim=0.
        collated[key] = torch.cat([item[key] for item in batch], dim=0)
    return collated


if __name__ == '__main__':
    from zero.expForwardKinematics.config.default import get_config
    from torch.utils.data import DataLoader, Dataset
    config_path = '/data/zero/zero/expForwardKinematics/config/DA3D.yaml'
    config = get_config(config_path)
    data_dir = config['TrainDataset']['data_dir']
    dataset = DatasetDA3D(config, data_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collect_fn)

    for i, batch in enumerate(loader):
        for key in batch.keys():
            print(key, batch[key].shape)
        break
