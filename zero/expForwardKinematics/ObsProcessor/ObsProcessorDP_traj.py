from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
import numpy as np
import torch
from copy import deepcopy as copy
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import *
from codebase.z_utils.open3d import *
from codebase.z_utils.idx_mask import *
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
    def obs_2_obs_raw(self, obs):
        if not isinstance(obs, list):
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

            out['rgb'] = np.stack(out['rgb'], axis=0)  # (T, N, H, W, 3)
            out['xyz'] = np.stack(out['xyz'], axis=0)  # (T, N, H, W, 3)
            out['eePose'] = np.stack(out['eePose'], axis=0)  # (T, A)
            out['JP'] = np.stack(out['JP'], axis=0)  # (T, A)

        else:
            '''
            data = obs_raw = {
            'key_frameids': [],
            'rgb': [],  # (T, N, H, W, 3)
            'xyz': [],  # (T, N, H, W, 3)
            'eePose': [],  # (T, A)
            'bbox': [],  # [T of dict]
            'pose': [],  # [T of dict]
            'JP': [],
             }
             '''

            rgb = copy(data['rgb'])
            xyz = copy(data['xyz'])
            JP_hist = copy(data['JP'])
            eePose_hist = copy(data['eePose'])

            out['rgb'] = rgb
            out['xyz'] = xyz
            out['eePose'] = eePose_hist
            out['JP'] = JP_hist

        return out

    @override
    def dynamic_process(self, data, *args, **kwargs):
        if self.train_flag:
            return self.dynamic_process_train(data, *args, **kwargs)
        else:
            return self.dynamic_process_eval(data, *args, **kwargs)

    def dynamic_process_train(self, data, *args, **kwargs):
        '''
        从数据中sample一个batch出来,先sample,后处理,这样,train,eval共用代码了。
        默认data中的数据都是numpy stack好的
        '''
        batch = {'obs': {}, 'action': None}
        episode_length = len(data['rgb'])
        start_frame = np.random.randint(0, episode_length - 1)  # 最后一个不选,此时,没有下一个动作

        index_hist = max(0, start_frame - (self.chunk_size - 1))  # 历史长度包括了现在
        index_futr = min(episode_length, start_frame + self.chunk_size)

        rgb_hist = copy(data['rgb'][index_hist:(start_frame + 1)][None, ...])  # (B,T,ncam,H,W,C)

        eePose_hist = copy(data['eePose'][index_hist:(start_frame + 1)][None, ...])  # (B,H,D)
        eePose_futr = copy(data['eePose'][(start_frame + 1):(index_futr + 1)][None, ...])

        JP_hist = copy(data['JP'][index_hist:(start_frame + 1)][None, ...])
        JP_futr = copy(data['JP'][(start_frame + 1):(index_futr + 1)][None, ...])

        if self.config['DP']['ActionHead']['action_mode'] == 'JP':
            obs_action = JP_hist
            gt_action = JP_futr
        elif self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            obs_action = eePose_hist
            gt_action = eePose_futr
        else:
            raise NotImplementedError('action mode not supported: {}'.format(self.config['DP']['ActionHead']['action_mode']))

        batch['obs'].update(self._dynamic_process_image(rgb_hist))
        batch['obs'].update(self._dynamic_process_obs_action(obs_action))
        batch['action'] = self._dynamic_process_gt_action(gt_action)

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

    def _dynamic_process_image(self, rgb):
        '''
        rgb: (T, N, H, W, 3)

        outs:{
            'image0': (1, T, N, H, W, 3),
            'image1': (1, T, N, H, W, 3),
            'image2': (1, T, N, H, W, 3),
            'image3': (1, T, N, H, W, 3),
        }
        '''
        image0 = rgb[:, :, 0, :, :, :].transpose(0, 1, 4, 2, 3) / 255.0
        image1 = rgb[:, :, 1, :, :, :].transpose(0, 1, 4, 2, 3) / 255.0
        image2 = rgb[:, :, 2, :, :, :].transpose(0, 1, 4, 2, 3) / 255.0
        image3 = rgb[:, :, 3, :, :, :].transpose(0, 1, 4, 2, 3) / 255.0

        # to tensor
        image0 = torch.from_numpy(image0).float()
        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()
        image3 = torch.from_numpy(image3).float()

        if image0.shape[1] < self.chunk_size:
            n_pad = self.chunk_size - image0.shape[1]
            image0 = torch.cat([image0[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image0], dim=1)
            image1 = torch.cat([image1[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image1], dim=1)
            image2 = torch.cat([image2[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image2], dim=1)
            image3 = torch.cat([image3[:, 0:1, :, :, :].repeat(1, n_pad, 1, 1, 1), image3], dim=1)

        return {
            'image0': image0,
            'image1': image1,
            'image2': image2,
            'image3': image3, }

    def _dynamic_process_obs_action(self, obs_action):
        '''
        obs action
        '''
        if self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            eePos = obs_action[:, :, :3]
            eeRot = obs_action[:, :, 3:7]
            eeOpen = obs_action[:, :, 7:8]

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

            return {
                'eePos': eePos,
                'eeRot': eeRot,
                'eeOpen': eeOpen,
            }
        elif self.config['DP']['ActionHead']['action_mode'] == 'JP':
            JP = obs_action
            JP = normalize_JP(JP)
            JP = torch.from_numpy(JP).float()

            if JP.shape[1] < self.chunk_size:
                n_pad = self.chunk_size - JP.shape[1]
                JP = torch.cat([JP[:, 0:1, :].repeat(1, n_pad, 1), JP], dim=1)

            return {
                'JP_hist': JP,
            }
        else:
            raise NotImplementedError('action mode not supported: {}'.format(self.config['DP']['ActionHead']['action_mode']))

    def _dynamic_process_gt_action(self, action):

        if self.config['DP']['ActionHead']['action_mode'] == 'eePose':
            act_pos = normalize_pos(action[:, :, :3])
            act_rot = self.norm_rot(action[:, :, 3:7])
            act_open = eePose[:, :, 7:8]
            eePose = np.concatenate([act_pos, act_rot, act_open], axis=-1)

            eePose = torch.from_numpy(eePose).float()
            if eePose.shape[1] < self.chunk_size:
                n_pad = self.chunk_size - eePose.shape[1]
                eePose = torch.cat([eePose, eePose[:, -1:, :].repeat(1, n_pad, 1)], dim=1)
            return eePose

        elif self.config['DP']['ActionHead']['action_mode'] == 'JP':

            JP_futr = normalize_JP(action)
            JP_futr = torch.from_numpy(JP_futr).float()

            if JP_futr.shape[1] < self.chunk_size:
                n_pad = self.chunk_size - JP_futr.shape[1]
                JP_futr = torch.cat([JP_futr, JP_futr[:, -1:, :].repeat(1, n_pad, 1)], dim=1)

            return JP_futr

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
