import re
import open3d as o3d
import numpy as np
import torch
from copy import deepcopy as copy
import einops
import pickle
import json
from numpy import array as npa
from pathlib import Path
import random
from typing import Dict, Optional, Sequence
from collections import defaultdict, Counter
from zero.expForwardKinematics.models.lotus.utils.robot_box import RobotBox
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase
from zero.z_utils.utilities_all import *
import torchvision.transforms.functional as transforms_f
from zero.expForwardKinematics.models.lotus.utils.rotation_transform import quaternion_to_discrete_euler, RotationMatrixTransform
from zero.expForwardKinematics.ReconLoss.ForwardKinematics import FrankaEmikaPanda
import torchvision.transforms as transforms
from codebase.z_utils.open3d import *
from codebase.z_utils.idx_mask import *
from codebase.z_utils.Rotation import quat2euler, euler2quat
from scipy.spatial.transform import Rotation as R
from typing_extensions import override
# --------------------------------------------------------------
# region DA3D
DA3D_Instr = Dict[str, Dict[int, torch.Tensor]]


class ObsProcessorDA3DWrapper(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config)
        self.config = config
        self.dataset_init_flag = False
        self.train_flag = train_flag

    @override
    def dataset_init(self, **kwargs):
        instructions = self.load_instructions('/data/zero/wrapper/3d_diffuser_actor/instructions/peract/instructions.pkl')
        self._instructions = instructions
        self.apply_cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
        self.apply_rgb = True
        self.apply_depth = False
        self.apply_pc = True
        self.image_size = (256, 256)

        self.data_container = {
            'gripper': [],
        }

    @override
    def denormalize_action(self, action) -> list:
        '''
        no change
        '''
        action = [action[0, i, :].cpu().detach().numpy() for i in range(action.shape[1])]
        return action

    @override
    @staticmethod
    def collate_fn(batch):
        collated = {}
        for key in batch[0]:
            # Concatenate the tensors from each dict in the batch along dim=0.
            try:
                collated[key] = torch.cat([item[key] for item in batch], dim=0)
            except:
                continue
        return collated

    @torch.no_grad()
    def eval_process(self, obs, taskvar):
        interpolation_length = 2
        rgbs, pcds, gripper = self.get_rgb_pcd_gripper_from_obs(obs)
        task_str, variation = taskvar.split('_peract+')
        instrs = self._instructions[task_str][int(variation)]
        instr = random.choice(instrs).unsqueeze(0)
        fake_traj = torch.full(
            [1, interpolation_length - 1, gripper.shape[-1]], 0
        ).to(rgbs.device)
        traj_mask = torch.full(
            [1, interpolation_length - 1], False
        ).to(rgbs.device)

        self.update_data_container('gripper', gripper)
        gripper = torch.stack(self.data_container['gripper'], dim=1)

        batch = {
            'trajectory': fake_traj,
            'trajectory_mask': traj_mask,
            'rgbs': rgbs[:, :, :3, :, :],
            'pcds': pcds,
            'instr': instr,
            'gripper': gripper[..., :7],
        }
        return batch

    # utils

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = self._transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = self._obs_to_attn(obs, cam)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action(self, obs):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        return state_dict, torch.from_numpy(action).float()

    @staticmethod
    def _transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
        apply_depth = len(obs_dict.get("depth", [])) > 0
        apply_pc = len(obs_dict["pc"]) > 0
        num_cams = len(obs_dict["rgb"])

        obs_rgb = []
        obs_depth = []
        obs_pc = []
        for i in range(num_cams):
            rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
            depth = (
                torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
                if apply_depth
                else None
            )
            pc = (
                torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
            )

            if augmentation:
                raise NotImplementedError()  # Deprecated

            # normalise to [-1, 1]
            rgb = rgb / 255.0
            rgb = 2 * (rgb - 0.5)

            obs_rgb += [rgb.float()]
            if depth is not None:
                obs_depth += [depth.float()]
            if pc is not None:
                obs_pc += [pc.float()]
        obs = obs_rgb + obs_depth + obs_pc
        return torch.cat(obs, dim=0)

    @staticmethod
    def _obs_to_attn(obs, camera):
        extrinsics_44 = torch.from_numpy(
            obs.misc[f"{camera}_camera_extrinsics"]
        ).float()
        extrinsics_44 = torch.linalg.inv(extrinsics_44)
        intrinsics_33 = torch.from_numpy(
            obs.misc[f"{camera}_camera_intrinsics"]
        ).float()
        intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
        gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
        gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
        points_cam_41 = extrinsics_44 @ gripper_pos_41

        proj_31 = intrinsics_34 @ points_cam_41
        proj_3 = proj_31.float().squeeze(1)
        u = int((proj_3[0] / proj_3[2]).round())
        v = int((proj_3[1] / proj_3[2]).round())

        return u, v

    @staticmethod
    def load_instructions(
        instructions: Optional[Path],
        tasks: Optional[Sequence[str]] = None,
        variations: Optional[Sequence[int]] = None,
    ) -> Optional[DA3D_Instr]:
        if instructions is not None:
            with open(instructions, "rb") as fid:
                data: DA3D_Instr = pickle.load(fid)
            if tasks is not None:
                data = {task: var_instr for task, var_instr in data.items() if task in tasks}
            if variations is not None:
                data = {
                    task: {
                        var: instr for var, instr in var_instr.items() if var in variations
                    }
                    for task, var_instr in data.items()
                }
            return data
        return None

    def update_data_container(self, name, value):
        length = len(self.data_container[name])
        H = 3  # TODO:horizon

        if length == H:
            self.data_container[name].pop(0)
            self.data_container[name].append(value)
        elif length == 0:
            [self.data_container[name].append(value)for _ in range(H)]
        else:
            raise ValueError(f"data_container {name} length is {length}, but it should be 0 or {H}.")


class ObsProcessorDA3D_Old(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.train_flag = train_flag

    def find_middle_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg', horizon=8):

        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    def static_process(self, data):
        out = {
            'xyz': [],
            'rgb': [],
            'eePose_hist': [],
            'eePose_futr': [],
            'JP_hist': [],
            'JP_futr': []
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
                action_next = copy(np.array(data['eePose'][i + 1], dtype=np.float64))
                action_path = copy(np.array(action_all[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

                open_all = np.array([a[7] for a in action_all])
                JP_all_copy = copy(JP_all)
                JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

                JP_curr = copy(np.array(JP_all_copy[data['key_frameids'][i]], dtype=np.float64))
                JP_next = copy(np.array(JP_all_copy[data['key_frameids'][i + 1]], dtype=np.float64))
                JP_path = copy(np.array(JP_all_copy[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))

                # eePose_hist
                if keyframe_id - 8 <= 1:
                    eePose_hist = [action_all[j] for j in range(keyframe_id)]
                    eePose_hist += [action_curr] * (8 - keyframe_id)

                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id)]
                    JP_hist += [JP_curr] * (8 - keyframe_id)
                else:
                    eePose_hist = [action_all[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                # eePose_futr
                eePose_futr, JP_futr = self.find_middle_actions(action_path, JP_path, sub_keyframe_dection_mode='avg')

                # concatenate
                eePose_hist = np.stack(eePose_hist, axis=0)
                eePose_futr = np.stack(eePose_futr, axis=0)
                JP_hist = np.stack(JP_hist, axis=0)
                JP_futr = np.stack(JP_futr, axis=0)
                # check & save
                assert np.allclose(action_curr, eePose_hist[-1])
                assert np.allclose(action_next, eePose_futr[-1])
                assert np.allclose(action_curr, eePose_hist[-1])

                assert np.allclose(JP_curr, JP_all_copy[keyframe_id])
                assert np.allclose(JP_next, JP_futr[-1])
                assert np.allclose(JP_curr, JP_hist[-1])

                out['rgb'].append(rgb)
                out['xyz'].append(xyz)
                out['eePose_hist'].append(eePose_hist)
                out['eePose_futr'].append(eePose_futr)
                out['JP_hist'].append(JP_hist)
                out['JP_futr'].append(JP_futr)
        else:
            rgb = copy(data['rgb'])
            xyz = copy(data['xyz'])
            eePose_hist = copy(data['eePose_hist_eval'])
            JP_hist = copy(data['JP_hist_eval'])

            out['rgb'].append(rgb[0])
            out['xyz'].append(xyz[0])
            out['JP_hist'].append(JP_hist)
            out['eePose_hist'].append(eePose_hist)
        return out

    def dynamic_process(self, data, taskvar):
        outs = {
            'rgb': None,
            'xyz': None,
            'instr': None,
            'JP_hist': None,
            'JP_futr': None,
            'eePose_hist': None,
            'eePose_futr': None
        }

        B = len(data['rgb'])
        H = self.config['DiffuserActor']['ActionHead']['horizon']
        rgb = torch.from_numpy(np.stack(copy(data['rgb']), axis=0))
        xyz = torch.from_numpy(np.stack(copy(data['xyz']), axis=0))
        JP_hist = torch.from_numpy(np.stack(copy(data['JP_hist']), axis=0))[:, :H, :]
        eePose_hist = torch.from_numpy(np.stack(copy(data['eePose_hist']), axis=0))[:, :H, :]
        # instruction = torch.tensor(np.stack(copy(data['txt_embed']), axis=0)).squeeze()
        instr = []
        for i in range(B):
            instr_s = random.choice(self.taskvar_instrs[taskvar])
            instr_s = copy(self.instr_embeds[instr_s])
            instr_s, mask = pad_clip_features([instr_s], 53)
            instr.append(instr_s)
        instr = torch.tensor(np.stack(instr, axis=0)).squeeze(0)

        # augmentation
        resized_dict = self._resize(rgb=rgb, xyz=xyz)
        rgb = resized_dict['rgb']
        xyz = resized_dict['xyz']

        rgb = einops.rearrange(rgb, 'bs ncam h w c-> bs ncam c h w')
        xyz = einops.rearrange(xyz, 'bs ncam h w c-> bs ncam c h w')

        # normalize
        rgb = (rgb.float() / 255.0) * 2 - 1
        JP_hist = normalize_JP(JP_hist)

        eePose_hist[:, :, :3] = normalize_pos(eePose_hist[:, :, :3])

        xyz = torch.permute(normalize_pos(torch.permute(xyz, [0, 1, 3, 4, 2])), [0, 1, 4, 2, 3])

        eePose_hist = convert_rot(eePose_hist)

        # 下面接 Policy的forward
        if self.train_flag:  # TODO: reorganize the code
            JP_futr = torch.from_numpy(np.stack(copy(data['JP_futr']), axis=0))[:, :H, :]
            eePose_futr = torch.from_numpy(np.stack(copy(data['eePose_futr']), axis=0))[:, :H, :]
            JP_futr = normalize_JP(JP_futr)
            eePose_futr[:, :, :3] = normalize_pos(eePose_futr[:, :, :3])
            eePose_futr = convert_rot(eePose_futr)
            outs['JP_futr'] = JP_futr.float()
            outs['eePose_futr'] = eePose_futr.float()

        # return
        outs['rgb'] = rgb.float()
        outs['xyz'] = xyz.float()
        outs['instr'] = instr.float()
        outs['JP_hist'] = JP_hist.float()
        outs['eePose_hist'] = eePose_hist.float()
        return outs

    def dataset_init(self):
        config = self.config
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self._resize = Resize(config['TrainDataset']['image_rescales'])

    @staticmethod
    def collate_fn(batch):
        collated = {}
        for key in batch[0]:
            # Concatenate the tensors from each dict in the batch along dim=0.
            try:
                collated[key] = torch.cat([item[key] for item in batch], dim=0)
            except:
                continue
        return collated


# endregion
# --------------------------------------------------------------
# region DP


class ObsProcessorDP(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.train_flag = train_flag

    @override
    def static_process(self, data):
        out = {
            'xyz': [],
            'rgb': [],
            'eePose_hist': [],
            'eePose_futr': [],
            'JP_hist': [],
            'JP_futr': []
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
                action_next = copy(np.array(data['eePose'][i + 1], dtype=np.float64))
                action_path = copy(np.array(action_all[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

                open_all = np.array([a[7] for a in action_all])
                JP_all_copy = copy(JP_all)
                JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

                JP_curr = copy(np.array(JP_all_copy[data['key_frameids'][i]], dtype=np.float64))
                JP_next = copy(np.array(JP_all_copy[data['key_frameids'][i + 1]], dtype=np.float64))
                JP_path = copy(np.array(JP_all_copy[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))

                # eePose_hist
                if keyframe_id - 8 <= 1:
                    eePose_hist = [action_all[j] for j in range(keyframe_id)]
                    eePose_hist += [action_curr] * (8 - keyframe_id)

                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id)]
                    JP_hist += [JP_curr] * (8 - keyframe_id)
                else:
                    eePose_hist = [action_all[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                # eePose_futr
                eePose_futr, JP_futr = self.find_middle_actions(action_path, JP_path)

                # concatenate
                eePose_hist = np.stack(eePose_hist, axis=0)
                eePose_futr = np.stack(eePose_futr, axis=0)
                JP_hist = np.stack(JP_hist, axis=0)
                JP_futr = np.stack(JP_futr, axis=0)
                # check & save
                assert np.allclose(action_curr, eePose_hist[-1])
                assert np.allclose(action_next, eePose_futr[-1])
                assert np.allclose(action_curr, eePose_hist[-1])

                assert np.allclose(JP_curr, JP_all_copy[keyframe_id])
                assert np.allclose(JP_next, JP_futr[-1])
                assert np.allclose(JP_curr, JP_hist[-1])

                out['rgb'].append(rgb)
                out['xyz'].append(xyz)
                out['eePose_hist'].append(eePose_hist)
                out['eePose_futr'].append(eePose_futr)
                out['JP_hist'].append(JP_hist)
                out['JP_futr'].append(JP_futr)
        else:
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
        batch = {}
        H = self.config['FK']['ActionHead']['horizon']
        rgb = torch.from_numpy(np.stack(copy(data['rgb']), axis=0))
        image0 = rgb[:, 0, :, :, :].permute(0, 3, 1, 2)
        image1 = rgb[:, 1, :, :, :].permute(0, 3, 1, 2)
        image2 = rgb[:, 2, :, :, :].permute(0, 3, 1, 2)
        image3 = rgb[:, 3, :, :, :].permute(0, 3, 1, 2)
        eePose = torch.from_numpy(np.stack(copy(data['eePose_hist']), axis=0))
        eePos = eePose[:, :, :3]
        eeRot = eePose[:, :, 3:7]
        eeOpen = eePose[:, :, 7:8]

        # normalize
        image0 = image0 / 255.0
        image1 = image1 / 255.0
        image2 = image2 / 255.0
        image3 = image3 / 255.0

        eePos = normalize_pos(eePos)
        eeRot = tensorfp32([quat2euler(eeRot[i]) for i in range(eeRot.shape[0])]) / 3.15

        batch = {
            'obs': {
                'image0': image0.unsqueeze(1),
                'image1': image1.unsqueeze(1),
                'image2': image2.unsqueeze(1),
                'image3': image3.unsqueeze(1),
                'eePos': eePos,
                'eeRot': eeRot,
                'eeOpen': eeOpen
            },
        }

        if self.train_flag:
            action = torch.from_numpy(np.stack(copy(data['eePose_futr']), axis=0))
            act_pos = normalize_pos(action[..., :3])
            act_rot = tensorfp32([quat2euler(action[i][..., 3:7]) for i in range(action.shape[0])]) / 3.15
            act_open = action[..., 7:8]
            action = torch.cat([act_pos, act_rot, act_open], dim=-1)
            batch['eePose'] = action[:, :H, :]

        return batch

    @override
    def dataset_init(self):
        config = self.config
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self._resize = Resize(config['TrainDataset']['image_rescales'])

    @override
    @staticmethod
    def collate_fn(batch):
        collated = {
            'obs': {},
            'eePose': None,
        }
        for key in batch[0]['obs'].keys():
            # Concatenate the tensors from each dict in the batch along dim=0.
            collated['obs'][key] = torch.cat([minibatch['obs'][key] for minibatch in batch], dim=0)
        try:
            collated['eePose'] = torch.cat([minibatch['eePose'] for minibatch in batch], dim=0)
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
        new_action = np.zeros((B, H, 8), dtype=np.float32)
        new_action[:, :, :3] = denormalize_pos(action[:, :, :3]).cpu().detach().numpy()
        angles = action[:, :, 3:6].cpu().detach().numpy() * 3.15

        angles = einops.rearrange(angles, 'b h d -> (b h) d')
        angles = euler2quat(angles)
        angles = einops.rearrange(angles, '(b h) d -> b h d', b=B, h=H)
        new_action[:, :, 3:7] = angles
        new_action[:, :, 7:8] = action[:, :, 6:7].cpu().detach().numpy()

        new_action = [new_action[0, i, :] for i in range(H)]

        return new_action
# endregion
# --------------------------------------------------------------
# region FK


class ObsProcessorFK(ObsProcessorRLBenchBase):
    def __init__(self, config, train_flag=True):
        '''
        simulator generate demo or obs
        1. demo or obs 2 obs_raw
        2. obs_static_process
        3. obs_dynamic_process
        4. collate_fn 2 batch
        '''
        super().__init__(config)
        self.config = config
        self.rotation_transform = RotationMatrixTransform()
        self.WORKSPACE = get_robot_workspace(real_robot=False, use_vlm=False)
        self.dataset_init_flag = False
        self.train_flag = train_flag
        self.franka = FrankaEmikaPanda()

    @override
    def static_process(self, obs_raw):
        '''
        obs_raw={
            'key_frameids': [],
            'rgb': [],
            'xyz': [],
            'eePose': [],
            'bbox': [],
            'pose': [],
            'sem': [],# 空的
            'eePose_all': [],
            'JP_all': [],
            'JP_curr_no_open':[],
            'JP_hist_eval':[],
        }
        '''

        obs_static_process = {
            'xyz': [],
            'rgb': [],
            'eePose_hist': [],
            'eePose_futr': [],
            'JP_hist': [],
            'JP_futr': [],
            'mask': [],  # mask for collision, only with true the point will be counted when calculating the collision loss
            'arm_links_info': [],
            'noncollision_mask': [],
        }

        # all_names = episode_path.split('/')
        # task_name = all_names[6]
        # variation_name = all_names[7].split('variation')[-1]
        # episode_name = all_names[9]
        # taskvar = f'{task_name}_peract+{variation_name}'

        '''
        1. remove outside workspace
        2. remove table
        3. voxelization
        '''

        if self.train_flag:
            num_keyframes_with_end = len(obs_raw['key_frameids'])
            num_keyframes = num_keyframes_with_end - 1
        else:
            num_keyframes_with_end = 1
            num_keyframes = 1

        VoxelGrid_list = []

        for t in range(num_keyframes_with_end):  # voxelize first
            arm_links_info = (obs_raw['bbox'][t], obs_raw['pose'][t])
            xyz = obs_raw['xyz'][t].reshape(-1, 3)
            rgb = obs_raw['rgb'][t].reshape(-1, 3)

            # 1. within workspace
            in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) &\
                (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) &\
                (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
            # 2. remove table
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
            xyz = xyz[in_mask]
            rgb = rgb[in_mask]

            # 3. voxelize
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd, _, trace = pcd.voxel_down_sample_and_trace(
            #     self.config.Dataset.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
            # )
            # xyz = np.asarray(pcd.points)
            # trace = np.array([v[0] for v in trace])
            # rgb = rgb[trace]

            # 3. remove robot get gripper idx
            JP_curr = copy(np.array(obs_raw['JP_curr_no_open'][t], dtype=np.float64))
            JP_curr = np.concatenate([JP_curr, np.array([obs_raw['eePose'][t][-1]])], axis=0)
            mask = self._rm_robot_by_JP(xyz, JP_curr)
            xyz = xyz[~mask]
            rgb = rgb[~mask]

            # 4. FK voxelization
            VoxelGrid = o3d.geometry.VoxelGrid()
            VoxelGrid.voxel_size = self.config.Dataset.voxel_size
            VoxelGrid.origin = np.array([0, 0, 0])
            # xyz(555,3)
            voxel_index_set = set()
            voxel_points_dict = {}

            # 算index
            for point_idx, xyz_s in enumerate(xyz):
                grid_index = tuple(np.floor((xyz_s - VoxelGrid.origin) / VoxelGrid.voxel_size).astype(int))
                voxel_index_set.add(grid_index)

                if grid_index not in voxel_points_dict:
                    voxel_points_dict[grid_index] = []
                voxel_points_dict[grid_index].append(point_idx)

            # 创建voxel,这里先不考虑augmentation的问题
            for voxel_idx in voxel_index_set:
                voxel_xyz = xyz[voxel_points_dict[voxel_idx]]
                voxel_rgb = rgb[voxel_points_dict[voxel_idx]]
                voxel_s = o3d.geometry.Voxel(grid_index=voxel_idx, color=voxel_rgb[0],)
                VoxelGrid.add_voxel(voxel_s)
            VoxelGrid_list.append(VoxelGrid)

        # test single voxelgrid
        # voxels = VoxelGrid.get_voxels()
        # for voxel in voxels:
        #     print(voxel.grid_index)
        # print('please delete this test')
        # /test
        if self.train_flag:
            for t in range(num_keyframes):
                vg_curr = VoxelGrid_list[t]
                vg_futr = VoxelGrid_list[t + 1]

                voxels_curr = vg_curr.get_voxels()
                voxels_futr = vg_futr.get_voxels()

                voxels_curr_dict = {tuple(voxel.grid_index): i for i, voxel in enumerate(voxels_curr)}
                voxels_futr_dict = {tuple(voxel.grid_index): i for i, voxel in enumerate(voxels_futr)}
                in_curr_not_in_futr = set(voxels_curr_dict.keys()) - set(voxels_futr_dict.keys())  # means moved voxels

                noncollision_idx = npa([voxels_curr_dict[voxel_idx]for voxel_idx in list(in_curr_not_in_futr)])  # important
                noncollision_mask = idx2mask(noncollision_idx, len(voxels_curr))  # 1 means noncollision, 0 means collision

                # voxel to pcd
                xyz = []
                rgb = []
                for voxel in voxels_curr:
                    grid_index = voxel.grid_index
                    center = voxel.grid_index * vg_curr.voxel_size + vg_curr.origin + vg_curr.voxel_size / 2
                    xyz.append(center)
                    rgb.append(voxel.color)
                xyz = npa(xyz)
                rgb = npa(rgb)

                # 此时noncollision_mask是this_xyz的mask

                # remove outliers
                _, mask = pcd_remove_outliers(xyz, nb_neighbors=10, std_ratio=2.0,)
                xyz = xyz[mask]
                rgb = rgb[mask]
                noncollision_mask = noncollision_mask[mask]  # mask是全部point中保留的东西
                noncollision_idx = mask2idx(noncollision_mask)  # mask是全部point中保留的东西

                # remove noncollision outliers
                xyz_noncollision = xyz[noncollision_mask]
                _, mask = pcd_remove_outliers(xyz_noncollision, nb_neighbors=10, std_ratio=2.0,)
                noncollision_idx = noncollision_idx[mask]  # mask 是noncollision中保留的东西
                noncollision_mask = idx2mask(noncollision_idx, len(xyz))  # mask是全部point中保留的东西, length 没变

                obs_static_process['xyz'].append(xyz)
                obs_static_process['rgb'].append(rgb)
                obs_static_process['noncollision_mask'].append(noncollision_mask)

                # print(1)
                # pcd_visualize(xyz, rgb)
                # pcd_visualize(xyz[noncollision_mask], rgb[noncollision_mask])
        else:
            vg_curr = VoxelGrid_list[0]
            voxels_curr = vg_curr.get_voxels()
            xyz = []
            rgb = []
            for voxel in voxels_curr:
                grid_index = voxel.grid_index
                center = voxel.grid_index * vg_curr.voxel_size + vg_curr.origin + vg_curr.voxel_size / 2
                xyz.append(center)
                rgb.append(voxel.color)
            xyz = npa(xyz)
            rgb = npa(rgb)
            # remove outliers
            _, mask = pcd_remove_outliers(xyz, nb_neighbors=10, std_ratio=2.0,)
            xyz = xyz[mask]
            rgb = rgb[mask]
            obs_static_process['xyz'].append(xyz)
            obs_static_process['rgb'].append(rgb)

        # only for train

        if self.train_flag:
            action_all = obs_raw['eePose_all']
            JP_all = obs_raw['JP_all']
            for t in range(num_keyframes_with_end):
                # copy
                keyframe_id = copy(np.array(obs_raw['key_frameids'][t], dtype=np.int16))
                eePose_curr = copy(np.array(obs_raw['eePose'][t], dtype=np.float64))
                eePose_next = copy(np.array(obs_raw['eePose'][t + 1], dtype=np.float64))
                eePose_path = copy(np.array(action_all[obs_raw['key_frameids'][t]:obs_raw['key_frameids'][t + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

                JP_all_copy = np.concatenate([copy(JP_all), np.array([a[7] for a in action_all])[:, None]], axis=1)

                JP_curr = copy(np.array(JP_all_copy[obs_raw['key_frameids'][t]], dtype=np.float64))
                JP_next = copy(np.array(JP_all_copy[obs_raw['key_frameids'][t + 1]], dtype=np.float64))
                JP_path = copy(np.array(JP_all_copy[obs_raw['key_frameids'][t]:obs_raw['key_frameids'][t + 1] + 1], dtype=np.float64))

                # action_history
                if keyframe_id - 8 <= 1:
                    eePose_hist = [action_all[j] for j in range(keyframe_id)]
                    eePose_hist += [eePose_curr] * (8 - keyframe_id)

                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id)]
                    JP_hist += [JP_curr] * (8 - keyframe_id)
                else:
                    eePose_hist = [action_all[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                    JP_hist = [JP_all_copy[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                # action_future
                eePose_futr, JP_futr = self.find_middle_actions(eePose_path, JP_path, sub_keyframe_dection_mode='avg')

                # concatenate
                eePose_hist = np.stack(eePose_hist, axis=0)
                eePose_futr = np.stack(eePose_futr, axis=0)

                JP_hist = np.stack(JP_hist, axis=0)
                JP_futr = np.stack(JP_futr, axis=0)

                # check & save
                assert np.allclose(eePose_curr, eePose_hist[-1])
                assert np.allclose(eePose_next, eePose_futr[-1])
                assert np.allclose(eePose_curr, eePose_hist[-1])

                assert np.allclose(JP_curr, JP_all_copy[keyframe_id])
                assert np.allclose(JP_next, JP_futr[-1])
                assert np.allclose(JP_curr, JP_hist[-1])

                obs_static_process['eePose_hist'].append(eePose_hist)
                obs_static_process['eePose_futr'].append(eePose_futr)

                obs_static_process['JP_hist'].append(JP_hist)
                obs_static_process['JP_futr'].append(JP_futr)
        else:
            JP_hist_eval = copy(np.array(obs_raw['JP_hist_eval'], dtype=np.float64))

            length = len(JP_hist_eval)

            JP_hist = []
            if length < 8:
                JP_hist = [JP_hist_eval[0]] * (8 - length)
                JP_hist.extend(JP_hist_eval)
            else:
                JP_hist = JP_hist_eval[-8:]
            assert len(JP_hist) == 8
            assert np.allclose(JP_curr, JP_hist[-1])
            JP_hist = np.stack(JP_hist, axis=0)
            obs_static_process['JP_hist'].append(JP_hist)

            # to make dataflow complete
            obs_static_process['JP_futr'].append([])
            obs_static_process['eePose_hist'].append([])
            obs_static_process['eePose_futr'].append([])
            obs_static_process['noncollision_mask'].append([])

        return obs_static_process

    @override
    def dynamic_process(self, data, taskvar):
        '''
        1. Downsample point cloud
        2. Normalize point cloud and rgb
        '''
        outs = {
            'pc_fts': [],
            'JP_hist': [],
            'JP_futr': [],
            'instr': [],
            'instr_mask': [],
            'noncollision_mask': [],
        }

        n_frames = len(data['rgb'])
        # dynamic process
        for i in range(n_frames):
            # 1. retrieve data
            xyz = npa(copy(data['xyz'][i]))
            rgb = npa(copy(data['rgb'][i]))
            JP_hist = npa(copy(data['JP_hist'][i]))
            height = np.expand_dims(npafp32(copy(xyz[:, 2])), axis=1)
            height = (height - self.TABLE_HEIGHT)

            choice = random.choice(self.taskvar_instrs[taskvar])
            instr, instr_mask = pad_clip_features([self.instr_embeds[choice]])
            instr = tensorfp32(instr).squeeze(0)
            instr_mask = torch.tensor(instr_mask, dtype=torch.bool).squeeze(0)
            if self.train_flag is True:
                noncollision_mask = npa(copy(data['noncollision_mask'][i]))
            if self.config['TrainDataset']['augmentation'] is True:
                # 2. downsample by number
                idx = pcd_random_downsample_by_num(xyz, rgb, num_points=self.num_points, return_idx=True)
                xyz = xyz[idx]
                rgb = rgb[idx]
                height = height[idx]
                if self.train_flag is True:
                    noncollision_mask = noncollision_mask[idx]
            else:
                pass
                # 3. normalize point cloud
            center = np.mean(xyz, 0)
            xyz = xyz - center
            rgb = (rgb / 255.0) * 2 - 1
            pc_fts = np.hstack([xyz, rgb, height])  # (N, 6)

            # normalize joint positions
            JP_hist = normalize_JP(JP_hist)
            if self.train_flag is True:
                JP_futr = tensorfp32(copy(data['JP_futr'][i]))
                JP_futr = normalize_JP(JP_futr)
                outs['JP_futr'].append(JP_futr)
                outs['noncollision_mask'].append(noncollision_mask)

            outs['pc_fts'].append(tensorfp32(pc_fts))
            outs['JP_hist'].append(tensorfp32(JP_hist))
            outs['instr'].append(tensorfp32(instr))
            outs['instr_mask'].append(instr_mask)
        return outs
        # from zero.expForwardKinematics.ReconLoss.ForwardKinematics import FrankaEmikaPanda
        # franka = FrankaEmikaPanda()
        # for JP in JP_futr:
        #     franka.visualize_pcd(xyz, rgb / 255, JP)

        # 暂时只要了 rgb,pcd,joint_position_history,joint_position_future和txt
    @override
    def dataset_init(self):
        self.taskvar_instrs = json.load(open(self.config.TRAIN_DATASET.taskvar_instr_file))
        self.instr_embeds = np.load(self.config.TRAIN_DATASET.instr_embed_file, allow_pickle=True).item()
        # 0.1 Downsample args
        self.num_points = self.config.TRAIN_DATASET.num_points

        # 0.2 shift and normalization
        self.xyz_shift = self.config.TRAIN_DATASET.xyz_shift
        self.xyz_norm = self.config.TRAIN_DATASET.xyz_norm

        # put together
        self.use_height = self.config.TRAIN_DATASET.use_height

        # augment & action head
        self.rot_type = self.config.TRAIN_DATASET.rot_type
        self.augment_xyz = self.config.TRAIN_DATASET.augment_pc
        self.aug_max_rot = np.deg2rad(self.config.TRAIN_DATASET.aug_max_rot)
        self.euler_resolution = self.config.TRAIN_DATASET.euler_resolution

        self.pos_type = self.config.TRAIN_DATASET.pos_type
        self.pos_bins = self.config.TRAIN_DATASET.pos_bins
        self.pos_bin_size = self.config.TRAIN_DATASET.pos_bin_size
        self.pos_heatmap_type = self.config.TRAIN_DATASET.pos_heatmap_type
        self.pos_heatmap_no_robot = self.config.TRAIN_DATASET.pos_heatmap_no_robot

        # 0.1. Load some pheripheral information
        self.real_robot = self.config.TRAIN_DATASET.real_robot
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=self.real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()
        self.dataset_init_flag = True

    @override
    @staticmethod
    def collate_fn(data):
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
    # private functions

    def _get_groundtruth_rotations(self, action,):
        gt_rots = torch.from_numpy(action.copy())   # quaternions
        rot_type = self.config.TRAIN_DATASET.rot_type
        euler_resolution = self.config.TRAIN_DATASET.euler_resolution
        if rot_type == 'euler':    # [-1, 1]
            gt_rots = self.rotation_transform.quaternion_to_euler(gt_rots[1:]) / 180.
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        elif rot_type == 'euler_disc':  # 3D
            gt_rots = [quaternion_to_discrete_euler(x, euler_resolution) for x in gt_rots[1:]]
            gt_rots = torch.from_numpy(np.stack(gt_rots + gt_rots[-1:]))
        elif rot_type == 'euler_delta':
            gt_eulers = self.rotation_transform.quaternion_to_euler(gt_rots)
            gt_rots = (gt_eulers[1:] - gt_eulers[:-1]) % 360
            gt_rots[gt_rots > 180] -= 360
            gt_rots = gt_rots / 180.
            gt_rots = torch.cat([gt_rots, torch.zeros(1, 3)], 0)
        elif rot_type == 'rot6d':
            gt_rots = self.rotation_transform.quaternion_to_ortho6d(gt_rots)
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        else:
            gt_rots = torch.cat([gt_rots, gt_rots[-1:]], 0)
        gt_rots = gt_rots.numpy()
        return gt_rots

    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False

        robot_box = RobotBox(
            arm_links_info, keep_gripper=keep_gripper,
            env_name='rlbench', selfgen=True
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask

    def _rotate_gripper(self, gripper_rot, angle):
        rot = R.from_euler('z', angle, degrees=False)
        gripper_rot = R.from_quat(gripper_rot)
        gripper_rot = (rot * gripper_rot).as_quat()
        return gripper_rot

    def _find_gt_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg'):
        if sub_keyframe_dection_mode == 'avg':
            indices = np.linspace(0, len(actions_path) - 1, self.config.horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
            gt_actions = [actions_path[i] for i in indices]
            gt_theta_actions = [theta_actions_path[i] for i in indices]
            return gt_actions, gt_theta_actions
        elif sub_keyframe_dection_mode == 'xyzpeak':
            NotImplementedError("XYZPEAK")

    def _dataset_init_FK(self):
        config = self.config
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self.num_points = config['TrainDataset']['num_points']
        self.TABLE_HEIGHT = get_robot_workspace(real_robot=False)['TABLE_HEIGHT']

    def _rm_robot_by_JP(self, xyz, JP):
        theta = JP - self.franka.JP_offset
        bbox_link, bbox_other = self.franka.theta2obbox(theta)
        bbox_all = bbox_link + bbox_other[:2]
        pcd_idx = get_robot_pcd_idx(xyz, bbox_all)
        return pcd_idx

    def get_uncollision_mask(self, xyz, JP):
        theta = JP - self.franka.JP_offset
        bbox_link, bbox_other = self.franka.theta2obbox(theta)
        gripper_idx = get_robot_pcd_idx(xyz, *[bbox_other[2:]])
        return gripper_idx

    def find_middle_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg', horizon=8):
        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    def within_workspace(self, xyz, rgb):
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) &\
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) &\
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        xyz = xyz[in_mask]
        rgb = rgb[in_mask]
        return xyz, rgb

    def remove_table(self, xyz, rgb):
        in_mask = xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT']
        xyz = xyz[in_mask]
        rgb = rgb[in_mask]
        return xyz, rgb

    def voxelize(self, xyz, rgb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.Dataset.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        return xyz, rgb
# endregion

# region utils


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


def tensorfp32(x):
    if torch.is_tensor(x):
        x = x.float()
    else:
        x = torch.tensor(x, dtype=torch.float32)
    return x


def npafp32(x):
    return np.array(x, dtype=np.float32)


def convert_gripper_pose_world_to_image(obs, camera: str):
    '''Convert the gripper pose from world coordinate system to image coordinate system.
    image[v, u] is the gripper location.
    '''
    extrinsics_44 = obs.misc[f"{camera}_camera_extrinsics"].astype(np.float32)
    extrinsics_44 = np.linalg.inv(extrinsics_44)

    intrinsics_33 = obs.misc[f"{camera}_camera_intrinsics"].astype(np.float32)
    intrinsics_34 = np.concatenate([intrinsics_33, np.zeros((3, 1), dtype=np.float32)], 1)

    gripper_pos_31 = obs.gripper_pose[:3].astype(np.float32)[:, None]
    gripper_pos_41 = np.concatenate([gripper_pos_31, np.ones((1, 1), dtype=np.float32)], 0)

    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31[:, 0]

    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v


def get_robot_pcd_idx(xyz, obbox):
    points = o3d.utility.Vector3dVector(xyz)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = points
    # o3d.visualization.draw_geometries([pcd, *obbox])
    robot_point_idx = set()
    for box in obbox:
        tmp = box.get_point_indices_within_bounding_box(points)
        robot_point_idx = robot_point_idx.union(set(tmp))
    robot_point_idx = np.array(list(robot_point_idx))
    mask = np.zeros(len(xyz), dtype=bool)
    mask[robot_point_idx] = True
    return mask


def get_robot_workspace(real_robot=False, use_vlm=False):
    if real_robot:
        # ur5 robotics room
        if use_vlm:
            TABLE_HEIGHT = 0.0  # meters
            X_BBOX = (-0.60, 0.2)        # 0 is the robot base
            Y_BBOX = (-0.54, 0.54)  # 0 is the robot base
            Z_BBOX = (-0.02, 0.75)      # 0 is the table
        else:
            TABLE_HEIGHT = 0.01  # meters
            X_BBOX = (-0.60, 0.2)        # 0 is the robot base
            Y_BBOX = (-0.54, 0.54)  # 0 is the robot base
            Z_BBOX = (0, 0.75)      # 0 is the table

    else:
        # rlbench workspace
        TABLE_HEIGHT = 0.7505  # meters

        X_BBOX = (-0.5, 1.5)    # 0 is the robot base
        Y_BBOX = (-1, 1)        # 0 is the robot base
        Z_BBOX = (0.2, 2)       # 0 is the floor

    return {
        'TABLE_HEIGHT': TABLE_HEIGHT,
        'X_BBOX': X_BBOX,
        'Y_BBOX': Y_BBOX,
        'Z_BBOX': Z_BBOX
    }


# endregion


if __name__ == '__main__':
    pass
