import zarr
import numpy as np
import pickle
import os
from zero.tmp.replay_buffer import ReplayBuffer
from typing import Optional


class SequenceSampler:
    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 sequence_length: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=dict(),
                 episode_mask: Optional[np.ndarray] = None,
                 ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert (sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = self.create_indices(episode_ends,
                                          sequence_length=sequence_length,
                                          pad_before=pad_before,
                                          pad_after=pad_after,
                                          episode_mask=episode_mask
                                          )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
                except Exception as e:
                    import pdb
                    pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

    @staticmethod
    def create_indices(
            episode_ends: np.ndarray, sequence_length: int,
            episode_mask: np.ndarray,
            pad_before: int = 0, pad_after: int = 0,
            debug: bool = True) -> np.ndarray:
        episode_mask.shape == episode_ends.shape
        pad_before = min(max(pad_before, 0), sequence_length - 1)
        pad_after = min(max(pad_after, 0), sequence_length - 1)

        indices = list()
        for i in range(len(episode_ends)):
            if not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i - 1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx

            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after

            # range stops one idx before end
            for idx in range(min_start, max_start + 1):
                buffer_start_idx = max(idx, 0) + start_idx
                buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
                start_offset = buffer_start_idx - (idx + start_idx)
                end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
                sample_start_idx = 0 + start_offset
                sample_end_idx = sequence_length - end_offset
                if debug:
                    assert (start_offset >= 0)
                    assert (end_offset >= 0)
                    assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
                indices.append([
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx])
        indices = np.array(indices)
        return indices


# Zarr 存储路径
zarr_path = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DP_traj/trajectory/put_groceries_100_zarr'

# 创建 ReplayBuffer (如果不存在则会创建)
replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')  # 'a' 模式表示追加数据
single_data = replay_buffer.get_episode(0)  # 获取第一个 episode 的数据

print("Single data:", single_data.keys())
print(replay_buffer.meta)


n_obs_steps = 4
sampler = SequenceSampler(
    replay_buffer=replay_buffer,
    sequence_length=5,
    pad_before=n_obs_steps - 1,
    pad_after=n_obs_steps - 1,
    keys=None,
    key_first_k=dict(),
    episode_mask=None
)


sequence = sampler.sample_sequence(0)
print(replay_buffer.n_episodes)
