from typing import Dict
import torch
import numpy as np
import copy
from zero.z_utils.coding import dict_apply
from zero.tmp.replay_buffer import ReplayBuffer
from zero.tmp.sampler import SequenceSampler, get_val_mask, downsample_mask, JianSampler

from copy import deepcopy
from zero.z_utils.normalizer_action import normalize_JP, normalize_pos, quat2ortho6D

from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP_traj_zarr import ObsProcessorDP_traj_zarr


class DatasetTmp(torch.utils.data.Dataset):
    def __init__(self,
                 config,
                 data_dir,
                 ObsProcessor: ObsProcessorDP_traj_zarr,
                 *args,
                 **kwargs,
                 ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.create_from_path(data_dir)

        self.sampler = JianSampler(
            replay_buffer=self.replay_buffer,
            To=8,
            Ta=8,
            # o_keys=['test'],
            # a_keys=['action'],
            o_keys=['image0', 'image1', 'image2', 'image3', 'eePose', 'JP'],
            a_keys=['eePose', 'JP'],
        )
        self.obs_processor = ObsProcessor(config)  # type: ObsProcessorDP_traj_zarr

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        batch = self.obs_processor.dynamic_process(sample)
        return batch


def test():
    zarr_path = "./1_Data/B_Preprocess/zarr/DP_traj_zarr/trajectory/test2/train"
    test_data = "./1_Data/test"
    dataset = DatasetTmp(None, data_dir=zarr_path)
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        pass


if __name__ == '__main__':
    test()
