import torch
import torch.nn as nn
import numpy as np
import collections
from zero.dataprocess.utils import convert_gripper_pose_world_to_image, keypoint_discovery


class ObsProcessorRLBenchBase:
    def __init__(self, config):
        self.config = config

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
            'xyz': [state_dict['xyz']],  # (T, N, H, W, 3)
            'eePose': [state_dict['gripper']],  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
            'JP_curr_no_open': positions,
        }
        return obs_raw

    def demo_2_obs_raw(self, demo):  # TODO: refine I/O variables name
        """Fetch the desired state based on the provided demo.
        :param obs: incoming obs
        :return: required observation (rgb, depth, xyz, gripper state)
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
            'xyz': state_dict_ls['xyz'],  # (T, N, H, W, 3)
            'eePose': action_ls,  # (T, A)
            'bbox': bbox,  # [T of dict]
            'pose': pose,  # [T of dict]
            'eePose_all': actions,
            'JP_all': positions,
        }
        return obs_raw

    def obs2dict(self, obs):
        apply_rgb = True
        apply_xyz = True
        apply_cameras = ("left_shoulder", "right_shoulder", "overhead", "front")
        apply_depth = True
        apply_sem = False
        gripper_pose = False
        # fetch state: (#cameras, H, W, C)
        state_dict = {"rgb": [], "depth": [], "xyz": [], "sem": []}
        for cam in apply_cameras:
            if apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                state_dict["rgb"] += [rgb]

            if apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                state_dict["depth"] += [depth]

            if apply_xyz:
                xyz = getattr(obs, "{}_point_cloud".format(cam))
                state_dict["xyz"] += [xyz]

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

    def static_process(self, obs_raw, **kwargs):
        raise NotImplementedError(
            "ObsProcessorBase: static_process is not implemented, please implement it in your own ObsProcessor class."
        )

    def dynamic_process(self, obs_raw, **kwargs):
        raise NotImplementedError(
            "ObsProcessorBase: dynamic_process is not implemented, please implement it in your own ObsProcessor class."
        )

    def dataset_init(self, **kwargs):
        raise NotImplementedError(
            "ObsProcessorBase: dataset_init is not implemented, please implement it in your own ObsProcessor class."
        )

    @staticmethod
    def collate_fn(self, **kwargs):
        raise NotImplementedError(
            "ObsProcessorBase: collate_fn is not implemented, please implement it in your own ObsProcessor class."
        )

    def denormalize_action(self, action):
        """
        Denormalize the action to the original space.
        :param action: (A, )
        :return: denormalized action
        """
        raise NotImplementedError(
            "ObsProcessorBase: denormalize_action is not implemented, please implement it in your own ObsProcessor class."
        )
