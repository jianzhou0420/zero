from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
import numpy as np
import torch
from copy import deepcopy

import json

from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import *

from codebase.z_utils.open3d import *
from codebase.z_utils.idx_mask import *
from codebase.z_utils.Rotation import quat2euler, euler2quat
from scipy.spatial.transform import Rotation as R
from typing_extensions import override
from zero.z_utils.normalizer_action import \
    (normalize_pos, denormalize_pos, quat2ortho6D, normalize_JP, denormalize_JP, ortho6d2quat,
        normalize_quat2euler, denormalize_quat2euler)

from rlbench.backend.observation import Observation

from zero.tmp.replay_buffer import ReplayBuffer
from zero.tmp.sampler import SequenceSampler
import einops


class ObsProcessorDP_traj_zarr(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.train_flag = train_flag

    @override
    def obs_2_obs_raw(self, obs):
        if isinstance(obs, Observation):
            obs = [obs]

        obs_raw = {
            'key_frameids': [],
            'rgb': [],  # (T, N, H, W, 3)
            'xyz': [],  # (T, N, H, W, 3)
            'eePose': [],  # (T, A)
            'bbox': [],  # [T of dict]
            'pose': [],  # [T of dict]
            'JP': [],
        }
        for s_obs in obs:
            key_frames = [0]
            state_dict = self.obs2dict(s_obs)

            action = np.concatenate([s_obs.gripper_pose, [s_obs.gripper_open]]).astype(np.float32)
            position = np.concatenate([s_obs.joint_positions, [s_obs.gripper_open]]).astype(np.float32)

            obs_raw['key_frameids'].append(key_frames)
            obs_raw['rgb'].append(state_dict['rgb'])  # (T, N, H, W, 3)
            obs_raw['xyz'].append(state_dict['xyz'])  # (T, N, H, W, 3)
            obs_raw['eePose'].append(state_dict['gripper'])  # (T, A)
            obs_raw['JP'].append(position)  # [T of dict]

        obs_raw['key_frameids'] = np.array(obs_raw['key_frameids'], dtype=np.int16)
        obs_raw['rgb'] = np.array(obs_raw['rgb'], dtype=np.float32)
        obs_raw['xyz'] = np.array(obs_raw['xyz'], dtype=np.float32)
        obs_raw['eePose'] = np.array(obs_raw['eePose'], dtype=np.float32)
        obs_raw['JP'] = np.array(obs_raw['JP'], dtype=np.float32)

        return obs_raw

    @override
    def static_process(self, data):
        '''
        no data normalization or augmentation
        '''
        out = {
            'image0': None,
            'image1': None,
            'image2': None,
            'image3': None,
            'eePose': None,
            'JP': None,
        }
        # figure out 多帧还是单帧,
        out['image0'] = einops.rearrange(data['rgb'][:, 0, :, :, :], 'T H W C -> T C H W')
        out['image1'] = einops.rearrange(data['rgb'][:, 1, :, :, :], 'T H W C -> T C H W')
        out['image2'] = einops.rearrange(data['rgb'][:, 2, :, :, :], 'T H W C -> T C H W')
        out['image3'] = einops.rearrange(data['rgb'][:, 3, :, :, :], 'T H W C -> T C H W')
        out['eePose'] = np.array(data['eePose_all'])  # (T, A)

        isopen = out['eePose'][:, -1]
        JP = np.array(data['JP_all'])  # (T, A)
        JP = np.concatenate([JP, isopen[:, None]], axis=-1)
        out['JP'] = JP

        return out

    @override
    def dynamic_process(self, data, *args, **kwargs):
        if self.train_flag:
            return self.dynamic_process_train(data, *args, **kwargs)
        else:
            return self.dynamic_process_eval(data, *args, **kwargs)

    def dynamic_process_train(self, sample, *args, **kwargs):
        '''

        '''

        image0 = deepcopy(sample['obs']['image0'])
        image1 = deepcopy(sample['obs']['image1'])
        image2 = deepcopy(sample['obs']['image2'])
        image3 = deepcopy(sample['obs']['image3'])

        eePose = deepcopy(sample['obs']['eePose'])

        eePos = eePose[:, :3]
        eeRot = eePose[:, 3:7]
        batch = {
            'obs': {
                'image0': image0,
                'image1': image1,
                'image2': image2,
                'image3': image3,
                'eePos': None,
            },
            'action': {
                'eePose': sample['action']['eePose'],
                'JP': sample['action']['JP'],
            },
        }
        return batch

        return batch

    def dynamic_process_eval(self, data, *args, **kwargs):
        '''
        从数据中sample一个batch出来
        '''
        batch = {'obs': {}, 'action': None}
        rgb_hist = copy(data['rgb'][None, ...])  # (B, T, N, H, W, C)
        eePose_hist = copy(data['eePose'][None, ...])  # (B, T, A)
        JP_hist = copy(data['JP'][None, ...])  # (B, T, A)
        if self.config['DP']['ActionHead']['action_mode'] == 'JP':
            obs_action = JP_hist
        elif self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            obs_action = eePose_hist

        batch['obs'].update(self._dynamic_process_image(rgb_hist))
        batch['obs'].update(self._dynamic_process_obs_action(obs_action))
        return batch

    @override
    @staticmethod
    def collate_fn(batch):
        collated = {
            'obs': {},
            'action': None,
        }
        for key in batch[0]['obs'].keys():
            # Concatenate the tensors from each dict in the batch along dim=0.
            collated['obs'][key] = torch.cat([minibatch['obs'][key] for minibatch in batch], dim=0)
        try:
            collated['action'] = torch.cat([minibatch['action'] for minibatch in batch], dim=0)
        except:
            pass
        return collated
