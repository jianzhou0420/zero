from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
import numpy as np
import torch
from copy import deepcopy as copy

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


class ObsProcessorDP_traj(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.train_flag = train_flag

    @override
    def static_process(self, data):
        out = {
            'xyz': [],
            'rgb': [],
            'eePose': [],
            'JP': [],
        }

        if self.train_flag:
            pass
            action_all = data['eePose_all']
            JP_all = data['JP_all']

            # save path
            num_frames = len(data['rgb']) - 1

            for i in range(num_frames):
                keyframe_id = copy(np.array(data['key_frameids'][i], dtype=np.int16))
                rgb = data['rgb'][i]
                xyz = data['xyz'][i]

                action_curr = copy(np.array(data['eePose'][i], dtype=np.float64))

                open_all = np.array([a[7] for a in action_all])
                JP_all_copy = copy(JP_all)
                JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

                JP_curr = copy(np.array(JP_all_copy[data['key_frameids'][i]], dtype=np.float64))

                out['rgb'].append(rgb)
                out['xyz'].append(xyz)
                out['eePose'].append(action_curr)
                out['JP'].append(JP_curr)

        else:  # TODO:
            rgb = copy(data['rgb'])
            xyz = copy(data['xyz'])
            eePose_hist = copy(data['eePose_hist_eval'])
            JP_hist = copy(data['JP_hist_eval'])

            out['rgb'].append(rgb[0])
            out['xyz'].append(xyz[0])
            out['JP_hist'].append(JP_hist)
            out['eePose_hist'].append(eePose_hist)
        return out

    @override
    def dynamic_process(self, data, *args, **kwargs):
        '''
        从数据中sample一个batch出来
        '''
        batch = {'obs': {}, 'action': None}
        episode_length = len(data['rgb'])
        start_frame = np.random.randint(0, episode_length - 1)  # 最后一个不选，此时，没有下一个动作

        index_hist = max(0, start_frame - (self.chunk_size - 1))  # 历史长度包括了现在
        index_futr = min(episode_length, start_frame + self.chunk_size)

        rgb = np.stack(copy(data['rgb']), axis=0)
        image0 = rgb[index_hist:(start_frame + 1), 0, :, :, :].transpose(0, 3, 1, 2) / 255.0
        image1 = rgb[index_hist:(start_frame + 1), 1, :, :, :].transpose(0, 3, 1, 2) / 255.0
        image2 = rgb[index_hist:(start_frame + 1), 2, :, :, :].transpose(0, 3, 1, 2) / 255.0
        image3 = rgb[index_hist:(start_frame + 1), 3, :, :, :].transpose(0, 3, 1, 2) / 255.0

        # to tensor
        image0 = torch.from_numpy(image0).float().unsqueeze(0)
        image1 = torch.from_numpy(image1).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).float().unsqueeze(0)
        image3 = torch.from_numpy(image3).float().unsqueeze(0)

        if image0.shape[1] < self.chunk_size:
            n_pad = self.chunk_size - image0.shape[1]
            image0 = torch.cat([image0[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image0], dim=1)
            image1 = torch.cat([image1[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image1], dim=1)
            image2 = torch.cat([image2[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image2], dim=1)
            image3 = torch.cat([image3[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image3], dim=1)

        batch['obs']['image0'] = image0
        batch['obs']['image1'] = image1
        batch['obs']['image2'] = image2
        batch['obs']['image3'] = image3
        # actions
        if self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            eePose = np.stack(copy(data['eePose'][index_hist:(start_frame + 1)]), axis=0)[None, ...]
            eePos = eePose[:, :, :3]
            eeRot = eePose[:, :, 3:7]
            eeOpen = eePose[:, :, 7:8]

            eePos = normalize_pos(eePos)
            eeRot = self.norm_rot(eeRot)

            eePos = torch.from_numpy(eePos).float()
            eeRot = torch.from_numpy(eeRot).float()
            eeOpen = torch.from_numpy(eeOpen).float()

            # check need padding or not, padding them to the same length with nearest neighbor
            if eePos.shape[1] < self.chunk_size:
                n_pad = self.chunk_size - eePos.shape[1]
                eePos = torch.cat([eePos[:, 0:1, :].repeat(1, n_pad, 1), eePos], dim=1)
                eeRot = torch.cat([eeRot[:, 0:1, :].repeat(1, n_pad, 1), eeRot], dim=1)
                eeOpen = torch.cat([eeOpen[:, 0:1, :].repeat(1, n_pad, 1), eeOpen], dim=1)

            batch['obs']['eePos'] = eePos
            batch['obs']['eeRot'] = eeRot
            batch['obs']['eeOpen'] = eeOpen

            if self.train_flag:
                try:
                    action = np.stack(copy(data['eePose'][(start_frame + 1):(index_futr + 1)]), axis=0)[None, ...]
                except:
                    pass
                act_pos = normalize_pos(action[:, :, :3])
                act_rot = self.norm_rot(action[:, :, 3:7])
                act_open = action[:, :, 7:8]
                action = np.concatenate([act_pos, act_rot, act_open], axis=-1)

                action = torch.from_numpy(action).float()
                if action.shape[1] < self.chunk_size:
                    n_pad = self.chunk_size - action.shape[1]
                    action = torch.cat([action, action[:, -1:, :].repeat(1, n_pad, 1)], dim=1)

                batch['action'] = action

        elif self.config['DP']['ActionHead']['action_mode'] == 'JP':
            JP_hist = np.stack(copy(data['JP'][index_hist:(start_frame + 1)]), axis=0)[None, ...]
            JP_hist = normalize_JP(JP_hist)
            JP_hist = torch.from_numpy(JP_hist).float()

            if JP_hist.shape[1] < self.chunk_size:
                n_pad = self.chunk_size - JP_hist.shape[1]
                JP_hist = torch.cat([JP_hist[:, 0:1, :].repeat(1, n_pad, 1), JP_hist], dim=1)

            batch['obs']['JP_hist'] = JP_hist

            if self.train_flag:
                JP_futr = np.stack(copy(data['JP'][(start_frame + 1):(index_futr + 1)]), axis=0)[None, ...]
                JP_futr = normalize_JP(JP_futr)
                JP_futr = torch.from_numpy(JP_futr).float()

                if JP_futr.shape[1] < self.chunk_size:
                    n_pad = self.chunk_size - JP_futr.shape[1]
                    JP_futr = torch.cat([JP_futr, JP_futr[:, -1:, :].repeat(1, n_pad, 1)], dim=1)

                batch['action'] = JP_futr
        return batch

    @override
    def dataset_init(self):
        config = self.config
        # self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        # self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self.chunk_size = config['DP']['ActionHead']['horizon']

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

    # utils
    def find_middle_actions(self, actions_path, theta_actions_path, horizon=8):
        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

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
