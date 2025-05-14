from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
import numpy as np
import torch
from copy import deepcopy
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import *
from codebase.z_utils.open3d import *
from codebase.z_utils.idx_mask import *
from typing_extensions import override
from zero.z_utils.normalizer_action import \
    (normalize_pos, denormalize_pos, quat2ortho6D, normalize_JP, denormalize_JP, ortho6d2quat,
        normalize_quat2euler, denormalize_quat2euler)

import einops


class ObsProcessorDP_traj_zarr_x0loss(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.train_flag = train_flag

    @override
    def obs_2_obs_raw(self, obs):
        if type(obs) is not list:
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
        out['eePose'] = np.array(data['eePose_all']) if self.train_flag else np.array(data['eePose'])  # (T, A)

        isopen = out['eePose'][:, -1]
        JP = np.array(data['JP_all']) if self.train_flag else np.array(data['JP'])  # (T, A)
        JP = np.concatenate([JP, isopen[:, None]], axis=-1)
        out['JP'] = JP

        return out

    @override
    def dynamic_process(self, data, *args, **kwargs):
        if self.train_flag:
            return self.dynamic_process_train(data, *args, **kwargs)
        else:
            return self.dynamic_process_eval(data, *args, **kwargs)

    def dynamic_process_image(self, sample, *args, **kwargs):
        image0 = deepcopy(sample['obs']['image0'])[None, ...]
        image1 = deepcopy(sample['obs']['image1'])[None, ...]
        image2 = deepcopy(sample['obs']['image2'])[None, ...]
        image3 = deepcopy(sample['obs']['image3'])[None, ...]
        image0 = torch.from_numpy(image0).float() / 255 * 2 - 1
        image1 = torch.from_numpy(image1).float() / 255 * 2 - 1
        image2 = torch.from_numpy(image2).float() / 255 * 2 - 1
        image3 = torch.from_numpy(image3).float() / 255 * 2 - 1
        return image0, image1, image2, image3

    def dynamic_process_obs_action(self, sample, *args, **kwargs):
        eePose = deepcopy(sample['obs']['eePose'])[None, ...]
        JP_hist = deepcopy(sample['obs']['JP'])[None, ...]
        eePos = eePose[:, :, :3]
        eeRot = eePose[:, :, 3:7]
        eeOpen = eePose[:, :, -1:]
        eePos = normalize_pos(eePos)
        eeRot = quat2ortho6D(eeRot)
        JP_hist = normalize_JP(JP_hist)
        eePos = torch.from_numpy(eePos).float()
        eeRot = torch.from_numpy(eeRot).float()
        eeOpen = torch.from_numpy(eeOpen).float()
        JP_hist = torch.from_numpy(JP_hist).float()
        return eePos, eeRot, eeOpen, JP_hist

    def _dynamic_process_gt_action(self, sample):
        # JP
        JP = deepcopy(sample['action']['JP'])[None, ...]
        JP_futr = normalize_JP(JP)
        JP_futr = torch.from_numpy(JP_futr).float()

        # futr
        eePose = deepcopy(sample['action']['eePose'])[None, ...]
        act_pos = normalize_pos(eePose[:, :, :3])
        act_rot = self.norm_rot(eePose[:, :, 3:7])
        act_open = eePose[:, :, 7:8]
        eePose_futr = np.concatenate([act_pos, act_rot, act_open], axis=-1)
        eePose_futr = torch.from_numpy(eePose_futr).float()
        return JP_futr, eePose_futr

    def dynamic_process_train(self, sample, *args, **kwargs):

        image0, image1, image2, image3 = self.dynamic_process_image(sample, *args, **kwargs)
        # 数据集里面是PosQuat
        eePos, eeRot, eeOpen, JP_hist = self.dynamic_process_obs_action(sample, *args, **kwargs)
        JP_futr, eePose_futr = self._dynamic_process_gt_action(sample)  # (1, H, D)
        # normalize
        # apply torch
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
            'action': JP_futr,
            'eePose': eePose_futr,
        }
        if self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            batch['obs'].pop('JP_hist')

        return batch

    def dynamic_process_eval(self, static_data, *args, **kwargs):
        '''
        data 是singleFrame static_data
        static_data = {
            'image0': None,
            'image1': None,
            'image2': None,
            'image3': None,
            'eePose': None,
            'JP': None,
        }
        '''
        image0 = deepcopy(static_data['image0'])[None, ...]
        image1 = deepcopy(static_data['image1'])[None, ...]
        image2 = deepcopy(static_data['image2'])[None, ...]
        image3 = deepcopy(static_data['image3'])[None, ...]
        eePose = deepcopy(static_data['eePose'])[None, ...]
        JP_hist = deepcopy(static_data['JP'])[None, ...][..., :-1]  # TODO:fix it
        eePos = eePose[:, :, :3]
        eeRot = eePose[:, :, 3:7]
        eeOpen = eePose[:, :, -1:]
        eePos = normalize_pos(eePos)
        eeRot = quat2ortho6D(eeRot)
        JP_hist = normalize_JP(JP_hist)
        image0 = torch.from_numpy(image0).float()
        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()
        image3 = torch.from_numpy(image3).float()
        eePos = torch.from_numpy(eePos).float()
        eeRot = torch.from_numpy(eeRot).float()
        eeOpen = torch.from_numpy(eeOpen).float()
        JP_hist = torch.from_numpy(JP_hist).float()
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
            'action': None
        }
        if self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            batch['obs'].pop('JP_hist')
        return batch

    @override
    def dataset_init(self, **kwargs):
        self.data_container = []

    @override
    @staticmethod
    def collate_fn(batch):
        collated = {
            'obs': {},
            'action': None,
            'eePose': None,
        }
        for key in batch[0]['obs'].keys():
            # Concatenate the tensors from each dict in the batch along dim=0.
            collated['obs'][key] = torch.cat([minibatch['obs'][key] for minibatch in batch], dim=0)
        try:
            collated['action'] = torch.cat([minibatch['action'] for minibatch in batch], dim=0)
        except:
            pass
        try:
            collated['eePose'] = torch.cat([minibatch['eePose'] for minibatch in batch], dim=0)
        except:
            pass
        return collated

    @override
    def denormalize_action(self, action: dict) -> list:
        '''
        assume action has shape (1, H, D), batch size, horizon, action dim
        '''
        action = action['action_pred']
        B, H, D = action.shape
        assert B == 1, 'batch size should be 1'
        action = action.cpu().detach().numpy()
        if self.config['DP']['ActionHead']['action_mode'] == 'JP':
            rlbench_action = denormalize_JP(action)
        elif self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            pos = denormalize_pos(action[:, :, :3])
            if self.config['DP']['ActionHead']['rot_norm_type'] == 'ortho6d':
                rot = ortho6d2quat(action[:, :, 3:9])
            elif self.config['DP']['ActionHead']['rot_norm_type'] == 'quat':
                rot = action[:, :, 3:7]
            elif self.config['DP']['ActionHead']['rot_norm_type'] == 'euler':
                rot = denormalize_quat2euler(action[:, :, 3:6])
            isopen = action[:, :, -1]
            rlbench_action = np.concatenate([pos, rot, isopen[..., None]], axis=-1)

        rlbench_action = [
            rlbench_action[0, j, :]for j in range(H)
        ]

        return rlbench_action

    def norm_rot(self, eeRot):
        if self.config['DP']['ActionHead']['rot_norm_type'] == 'ortho6d':
            eeRot = quat2ortho6D(eeRot)
        elif self.config['DP']['ActionHead']['rot_norm_type'] == 'euler':
            eeRot = normalize_quat2euler(eeRot)
        elif self.config['DP']['ActionHead']['rot_norm_type'] == 'quat':
            raise NotImplementedError('quat norm not implemented')

        return eeRot
