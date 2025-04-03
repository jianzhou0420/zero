from tqdm import tqdm
import re
import random
import copy
import json
import pickle
import os
from zero.expAugmentation.ObsProcessor.ObsProcessorBase import ObsProcessorBase
from zero.dataprocess.utils import convert_gripper_pose_world_to_image, keypoint_discovery
from zero.expAugmentation.models.lotus.utils.rotation_transform import quaternion_to_discrete_euler, RotationMatrixTransform
import collections
import numpy as np
from zero.z_utils.utilities_all import pad_clip_features
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

    def get_obs_raw_sample(self,):
        obs_raw = {
            'key_frameids': [],
            'rgb': [],  # (T, N, H, W, 3)
            'pc': [],  # (T, N, H, W, 3)
            'action': [],  # (T, A)
            'bbox': [],  # [T of dict]
            'pose': [],  # [T of dict]
            'sem': [],  # (T, N, H, W, 3)
            'actions_all': [],
            'positions_all': [],
        }
        return obs_raw
    # region obs_raw

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
            'rgb': state_dict['rgb'],  # (T, N, H, W, 3)
            'pc': state_dict['pc'],  # (T, N, H, W, 3)
            'action': state_dict['gripper'],  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
            'sem': state_dict['sem'],  # (T, N, H, W, 3)
        }
        return obs_raw

    def demo_2_obs_raw(self, demo):
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
            'positions_all': positions,
        }
        return obs_raw

    def raw_data_2_static_process(self, raw_data):

        pass

    def obs2dict(self, obs):
        apply_rgb = True
        apply_pc = True
        apply_cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
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

    # endregion

    def find_middle_actions(self, actions_path, theta_actions_path, sub_keyframe_dection_mode='avg', horizon=8):

        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions

    def static_process_fk(self, root_dir, task, variation, episode):
        out = {
            'rgb': [],
            'pcd': [],
            'txt_embed': [],
            'action_history': [],
            'action_future': [],
            'joint_position_history': [],
            'joint_position_future': []
        }

        # region 3.1 Path & Load
        data_folder = os.path.join(root_dir, task, variation, 'episodes', episode)
        with open(os.path.join(data_folder, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(data_folder, 'actions_all.pkl'), 'rb') as f:
            action_all = pickle.load(f)
        with open(os.path.join(data_folder, 'positions_all.pkl'), 'rb') as f:
            joint_position_all = pickle.load(f)

        # save path
        save_root = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DA3D'
        save_folder = os.path.join(save_root, task, variation, 'episodes', episode)
        num_frames = len(data['rgb']) - 1

        # load instructions
        if not hasattr(self, 'taskvar_instrs'):
            self.taskvar_instrs = json.load(open('/data/zero/assets/taskvars_instructions_peract.json'))
            self.instr_embeds = np.load('/data/zero/assets/instr_embeds_clip.npy', allow_pickle=True).item()

        taskvar = task + '_' + 'peract+' + variation.split('variation')[1]
        # endregion

        for i in range(num_frames):
            keyframe_id = copy.deepcopy(np.array(data['key_frameids'][i], dtype=np.int16))
            rgb = data['rgb'][i]
            pcd = data['pc'][i]

            action_curr = copy.deepcopy(np.array(data['action'][i], dtype=np.float64))
            action_next = copy.deepcopy(np.array(data['action'][i + 1], dtype=np.float64))
            action_path = copy.deepcopy(np.array(action_all[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))  # 这里加一是为了包含下一个关键帧

            open_all = np.array([a[7] for a in action_all])
            JP_all_copy = copy.deepcopy(joint_position_all)
            JP_all_copy = np.concatenate([JP_all_copy, open_all[:, None]], axis=1)

            joint_position_curr = copy.deepcopy(np.array(JP_all_copy[data['key_frameids'][i]], dtype=np.float64))
            joint_position_next = copy.deepcopy(np.array(JP_all_copy[data['key_frameids'][i + 1]], dtype=np.float64))
            joint_position_path = copy.deepcopy(np.array(JP_all_copy[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float64))

            # action_history
            if keyframe_id - 8 <= 1:
                action_history = [action_all[j] for j in range(keyframe_id)]
                action_history += [action_curr] * (8 - keyframe_id)

                joint_position_history = [JP_all_copy[j] for j in range(keyframe_id)]
                joint_position_history += [joint_position_curr] * (8 - keyframe_id)
            else:
                action_history = [action_all[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
                joint_position_history = [JP_all_copy[j] for j in range(keyframe_id - 7, keyframe_id + 1)]
            # action_future
            action_future, joint_position_future = self.find_middle_actions(action_path, joint_position_path, sub_keyframe_dection_mode='avg')

            instr = random.choice(self.taskvar_instrs[taskvar])
            instr_embed = copy.deepcopy(self.instr_embeds[instr])
            instr_embed = pad_clip_features([instr_embed], 77)
            # concatenate
            action_history = np.stack(action_history, axis=0)
            action_future = np.stack(action_future, axis=0)
            joint_position_history = np.stack(joint_position_history, axis=0)
            joint_position_future = np.stack(joint_position_future, axis=0)
            # check & save
            assert np.allclose(action_curr, action_history[-1])
            assert np.allclose(action_next, action_future[-1])
            assert np.allclose(action_curr, action_history[-1])

            assert np.allclose(joint_position_curr, JP_all_copy[keyframe_id])
            assert np.allclose(joint_position_next, joint_position_future[-1])
            assert np.allclose(joint_position_curr, joint_position_history[-1])

            out['rgb'].append(rgb)
            out['pcd'].append(pcd)
            out['txt_embed'].append(instr_embed)

            out['action_history'].append(action_history)
            out['action_future'].append(action_future)

            out['joint_position_history'].append(joint_position_history)
            out['joint_position_future'].append(joint_position_future)

        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, 'data.pkl'), 'wb') as f:
            pickle.dump(out, f)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


data_dir = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/2000demo_put_groceries/train/904744'
tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
obs_processor = ObsProcessorDA3D(config=None)

for i, task in enumerate(tasks_all):
    variations = sorted(os.listdir(os.path.join(data_dir, task)), key=natural_sort_key)
    for j, variation in enumerate(variations):
        episodes = sorted(os.listdir(os.path.join(data_dir, task, variation, 'episodes')), key=natural_sort_key)
        for k, episode in tqdm(enumerate(episodes)):
            obs_processor.static_process_fk(data_dir, task, variation, episode)
