from typing import Dict
import torch
import numpy as np
import copy
from zero.z_utils.coding import dict_apply
from zero.tmp.replay_buffer import ReplayBuffer
from zero.tmp.sampler import SequenceSampler, get_val_mask, downsample_mask, JianSampler

from copy import deepcopy
from zero.z_utils.normalizer_action import normalize_JP, normalize_pos, quat2ortho6D


class DatasetTmp(torch.utils.data.Dataset):
    def __init__(self,
                 config,
                 data_dir,
                 *args,
                 **kwargs,
                 ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(data_dir)

        self.sampler = JianSampler(
            replay_buffer=self.replay_buffer,
            To=8,
            Ta=8,
            # o_keys=['test'],
            # a_keys=['action'],
            o_keys=['image0', 'image1', 'image2', 'image3', 'eePose', 'JP'],
            a_keys=['eePose', 'JP'],
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        # currently dont use obs_processor
        # 数据集里面是PosQuat
        image0 = deepcopy(sample['obs']['image0'])[None, ...]
        image1 = deepcopy(sample['obs']['image1'])[None, ...]
        image2 = deepcopy(sample['obs']['image2'])[None, ...]
        image3 = deepcopy(sample['obs']['image3'])[None, ...]
        eePose = deepcopy(sample['obs']['eePose'])[None, ...]
        JP_hist = deepcopy(sample['obs']['JP'])[None, ...]
        JP_futr = deepcopy(sample['action']['JP'])[None, ...]
        eePos = eePose[:, :, :3]
        eeRot = eePose[:, :, 3:7]
        eeOpen = eePose[:, :, -1:]

        # normalize
        eePos = normalize_pos(eePos)
        eeRot = quat2ortho6D(eeRot)
        JP_hist = normalize_JP(JP_hist)
        JP_futr = normalize_JP(JP_futr)

        # apply torch
        image0 = torch.from_numpy(image0).float()
        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()
        image3 = torch.from_numpy(image3).float()
        eePos = torch.from_numpy(eePos).float()
        eeRot = torch.from_numpy(eeRot).float()
        eeOpen = torch.from_numpy(eeOpen).float()
        JP_hist = torch.from_numpy(JP_hist).float()
        JP_futr = torch.from_numpy(JP_futr).float()
        batch = {
            'obs': {
                'image0': image0,
                'image1': image1,
                'image2': image2,
                'image3': image3,
                'eePos': eePos,
                'eeRot': eeRot,
                'eeOpen': eeOpen,
                'JP_hist': JP_hist,
            },
            'action': JP_futr
        }
        return batch


def test():
    zarr_path = "/media/jian/ssd4t/zero/1_Data/B_Preprocess/zarr/DP_traj_zarr/trajectory/test2/train"
    test_data = "/media/jian/ssd4t/zero/1_Data/test"
    dataset = DatasetTmp(None, data_dir=zarr_path)
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        pass


if __name__ == '__main__':
    test()
