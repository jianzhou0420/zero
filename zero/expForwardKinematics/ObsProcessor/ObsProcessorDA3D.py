import einops
from tqdm import tqdm
import re
import random
import copy
import json
import pickle
import torch
import os
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorBase
from zero.dataprocess.utils import convert_gripper_pose_world_to_image, keypoint_discovery
from zero.expForwardKinematics.models.lotus.utils.rotation_transform import quaternion_to_discrete_euler, RotationMatrixTransform
import collections
import numpy as np
from zero.z_utils.utilities_all import pad_clip_features, normalize_JP, normalize_pos, convert_rot, gripper_loc_bounds
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
# ---------------------------------------------------------------
# region 0.Some tools


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
# ---------------------------------------------------------------
# region 1. ObsProcessorDA3D


class ObsProcessorDA3D(ObsProcessorBase):
    def __init__(self, config, train_flag=True):
        super().__init__(config,)
        self.config = config
        self.gripper_loc_bounds = gripper_loc_bounds
        self.train_flag = train_flag

    def obs_2_obs_raw(self, obs):
        key_frames = [0]
        state_dict = self.obs2dict(obs)

        bbox = []
        pose = []

        single_bbox = dict()
        single_pose = dict()
        for key, value in obs.misc.items():
            if key.split('_')[-1] == 'bbox':
                single_bbox[key.split('_bbox')[0]] = value
            if key.split('_')[-1] == 'pose':
                single_pose[key.split('_pose')[0]] = value
        bbox.append(single_bbox)
        pose.append(single_pose)

        actions = []
        positions = []
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)
        position = obs.joint_positions
        actions.append(action)
        positions.append(position)
        obs_raw = {
            'key_frameids': key_frames,
            'rgb': [state_dict['rgb']],  # (T, N, H, W, 3)
            'pc': [state_dict['pc']],  # (T, N, H, W, 3)
            'action': [state_dict['gripper']],  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
            'JP_curr_no_open': positions,
        }
        return obs_raw

    def demo_2_obs_raw(self, demo):  # TODO: refine I/O variables name
        """Fetch the desired state based on the provided demo.
        :param obs: incoming obs
        :return: required observation (rgb, depth, pc, gripper state)
        """

        key_frames = keypoint_discovery(demo)
        key_frames.insert(0, 0)

        state_dict_ls = collections.defaultdict(list)
        for f in key_frames:
            state_dict = self.obs2dict(demo._observations[f])
            for k, v in state_dict.items():
                if len(v) > 0:
                    # rgb: (N: num_of_cameras, H, W, C); gripper: (7+1, )
                    state_dict_ls[k].append(v)

        for k, v in state_dict_ls.items():
            state_dict_ls[k] = np.stack(v, 0)  # (T, N, H, W, C)

        action_ls = state_dict_ls['gripper']  # (T, 7+1)
        del state_dict_ls['gripper']

        # return demo, key_frames, state_dict_ls, action_ls

        gripper_pose = []
        for key_frameid in key_frames:
            gripper_pose.append(demo[key_frameid].gripper_pose)

        # get bbox and poses of each link
        bbox = []
        pose = []
        for key_frameid in key_frames:
            single_bbox = dict()
            single_pose = dict()
            for key, value in demo[key_frameid].misc.items():
                if key.split('_')[-1] == 'bbox':
                    single_bbox[key.split('_bbox')[0]] = value
                if key.split('_')[-1] == 'pose':
                    single_pose[key.split('_pose')[0]] = value
            bbox.append(single_bbox)
            pose.append(single_pose)

        # get actions_all
        # get positions_all
        actions = []
        positions = []
        for obs in demo._observations:
            action = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)
            position = obs.joint_positions
            actions.append(action)
            positions.append(position)

        obs_raw = {
            'key_frameids': key_frames,
            'rgb': state_dict_ls['rgb'],  # (T, N, H, W, 3)
            'pc': state_dict_ls['pc'],  # (T, N, H, W, 3)
            'action': action_ls,  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
            'sem': state_dict_ls['sem'],  # (T, N, H, W, 3)
            'actions_all': actions,
            'joint_position_all': positions,
        }
        return obs_raw

    def raw_data_2_static_process(self, raw_data):

        pass

    def obs2dict(self, obs):
        apply_rgb = True
        apply_pc = True
        apply_cameras = ("left_shoulder", "right_shoulder", "overhead", "front")
        apply_depth = True
        apply_sem = False
        gripper_pose = False
        # fetch state: (#cameras, H, W, C)
        state_dict = {"rgb": [], "depth": [], "pc": [], "sem": []}
        for cam in apply_cameras:
            if apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["pc"] += [pc]

            if apply_sem:
                sem = getattr(obs, "{}_mask".format(cam))
                state_dict["sem"] += [sem]

        # fetch gripper state (3+4+1, )
        gripper = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(
            np.float32
        )
        state_dict["gripper"] = gripper

        if gripper_pose:
            gripper_imgs = np.zeros(
                (len(apply_cameras), 1, 128, 128), dtype=np.float32
            )
            for i, cam in enumerate(apply_cameras):
                u, v = convert_gripper_pose_world_to_image(obs, cam)
                if u > 0 and u < 128 and v > 0 and v < 128:
                    gripper_imgs[i, 0, v, u] = 1
            state_dict["gripper_imgs"] = gripper_imgs

        return state_dict

    def find_middle_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg', horizon=8):

        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    def static_process_DA3D(self, data):
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
            action_all = data['actions_all']
            joint_position_all = data['joint_position_all']

            # save path
            num_frames = len(data['rgb']) - 1

            for i in range(num_frames):
                keyframe_id = copy.deepcopy(np.array(data['key_frameids'][i], dtype=np.int16))
                rgb = data['rgb'][i]
                xyz = data['pc'][i]

                action_curr = copy.deepcopy(np.array(data['action'][i], dtype=np.float64))
                action_next = copy.deepcopy(np.array(data['action'][i + 1], dtype=np.float64))
                action_path = copy.deepcopy(np.array(action_all[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

                open_all = np.array([a[7] for a in action_all])
                JP_all_copy = copy.deepcopy(joint_position_all)
                JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

                JP_curr = copy.deepcopy(np.array(JP_all_copy[data['key_frameids'][i]], dtype=np.float64))
                JP_next = copy.deepcopy(np.array(JP_all_copy[data['key_frameids'][i + 1]], dtype=np.float64))
                JP_path = copy.deepcopy(np.array(JP_all_copy[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))

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
            rgb = copy.deepcopy(data['rgb'])
            xyz = copy.deepcopy(data['pc'])
            eePose_hist = copy.deepcopy(data['eePose_hist_eval'])
            JP_hist = copy.deepcopy(data['JP_hist_eval'])

            out['rgb'].append(rgb[0])
            out['xyz'].append(xyz[0])
            out['JP_hist'].append(JP_hist)
            out['eePose_hist'].append(eePose_hist)
        return out

    def dynamic_process_DA3D(self, data, taskvar):
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
        rgb = torch.from_numpy(np.stack(copy.deepcopy(data['rgb']), axis=0))
        xyz = torch.from_numpy(np.stack(copy.deepcopy(data['xyz']), axis=0))
        JP_hist = torch.from_numpy(np.stack(copy.deepcopy(data['JP_hist']), axis=0))
        eePose_hist = torch.from_numpy(np.stack(copy.deepcopy(data['eePose_hist']), axis=0))
        # instruction = torch.tensor(np.stack(copy.deepcopy(data['txt_embed']), axis=0)).squeeze()
        instr = []
        for i in range(B):
            instr_s = random.choice(self.taskvar_instrs[taskvar])
            instr_s = copy.deepcopy(self.instr_embeds[instr_s])
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
            JP_futr = torch.from_numpy(np.stack(copy.deepcopy(data['JP_futr']), axis=0))
            eePose_futr = torch.from_numpy(np.stack(copy.deepcopy(data['eePose_futr']), axis=0))
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

    def _dataset_init_DA3D(self):
        config = self.config
        self.taskvar_instrs = json.load(open(config['TrainDataset']['taskvar_instr_file']))
        self.instr_embeds = np.load(config['TrainDataset']['instr_embed_file'], allow_pickle=True).item()
        self._resize = Resize(config['TrainDataset']['image_rescales'])

        pass

    @staticmethod
    def collect_fn(batch):
        collated = {}
        for key in batch[0]:
            # Concatenate the tensors from each dict in the batch along dim=0.
            try:
                collated[key] = torch.cat([item[key] for item in batch], dim=0)
            except:
                continue
        return collated

# endregion
# -------------------------------------------------------------------------------
# region utils


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


def static_process_data():
    data_dir = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/20demo_put_groceries/train/520837'
    tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
    obs_processor = ObsProcessorDA3D(config=None)
    save_root = '/data/zero/1_Data/B_Preprocess/DA3D'

    for i, task in enumerate(tasks_all):
        variations = sorted(os.listdir(os.path.join(data_dir, task)), key=natural_sort_key)
        for j, variation in enumerate(variations):
            episodes = sorted(os.listdir(os.path.join(data_dir, task, variation, 'episodes')), key=natural_sort_key)
            for k, episode in tqdm(enumerate(episodes)):
                outs = obs_processor.static_process_DA3D(data_dir, task, variation, episode)
                save_folder = os.path.join(save_root, task, variation, 'episodes', episode)
                os.makedirs(save_folder, exist_ok=True)
                with open(os.path.join(save_folder, 'data.pkl'), 'wb') as f:
                    pickle.dump(outs, f)


if __name__ == '__main__':
    static_process_data()
