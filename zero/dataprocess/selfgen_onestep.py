from rlbench.tasks import *
from datetime import datetime

import time
from rlbench.demo import Demo
from typing import List, Dict, Optional, Sequence, Tuple, TypedDict, Union, Any
import os
from multiprocessing import Process, Manager, Semaphore

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task
from tqdm import tqdm
import os
import pickle
import json
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np
import random
import collections


import sys
from typing import List, Tuple

import numpy as np

import torch
import einops
import json
from scipy.spatial.transform import Rotation as R

from zero.dataprocess.utils import convert_gripper_pose_world_to_image, keypoint_discovery


def get_obs(obs):
    """Fetch the desired state based on the provided demo.
    :param obs: incoming obs
    :return: required observation (rgb, depth, pc, gripper state)
    """
    apply_rgb = True
    apply_pc = True
    apply_cameras = ("left_shoulder", "right_shoulder", "wrist", "front")
    apply_depth = True
    apply_sem = True
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


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_observation(task_str: str, variation: int, episode: int, env):
    demo = env.get_demo(task_str, variation, episode)

    key_frames = keypoint_discovery(demo)
    key_frames.insert(0, 0)

    state_dict_ls = collections.defaultdict(list)
    for f in key_frames:
        state_dict = get_obs(demo._observations[f])
        for k, v in state_dict.items():
            if len(v) > 0:
                # rgb: (N: num_of_cameras, H, W, C); gripper: (7+1, )
                state_dict_ls[k].append(v)

    for k, v in state_dict_ls.items():
        state_dict_ls[k] = np.stack(v, 0)  # (T, N, H, W, C)

    action_ls = state_dict_ls['gripper']  # (T, 7+1)
    del state_dict_ls['gripper']

    return demo, key_frames, state_dict_ls, action_ls


def post_process_demo(demo, example_path):
    # Save image data first, and then None the image data, and pickle
    cameras = ['left_shoulder', 'right_shoulder', 'wrist', 'front']
    key_frames = keypoint_discovery(demo)
    key_frames.insert(0, 0)

    state_dict_ls = collections.defaultdict(list)
    for f in key_frames:
        state_dict = get_obs(demo._observations[f])
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

    outs = {
        'key_frameids': key_frames,
        'rgb': state_dict_ls['rgb'],  # (T, N, H, W, 3)
        'pc': state_dict_ls['pc'],  # (T, N, H, W, 3)
        'action': action_ls,  # (T, A)
        'bbox': bbox,  # [T of dict]
        'pose': pose,  # [T of dict]
        'sem': state_dict_ls['sem'],  # (T, N, H, W, 3)
    }
    check_and_make(example_path)
    with open(os.path.join(example_path, 'data.pkl'), 'wb') as f:
        pickle.dump(outs, f)

    # save actions
    actions = []
    for obs in demo._observations:
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]]).astype(np.float32)
        actions.append(action)
    with open(os.path.join(example_path, 'actions_all.pkl'), 'wb') as f:
        pickle.dump(actions, f)


def run(task, semaphore, config, pbar):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    # np.random.seed(None)
    with semaphore:
        # region 0.seed
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        # endregion
        # region 1.config
        img_size = list(map(int, config['image_size']))

        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.overhead_camera.rgb = False
        obs_config.overhead_camera.depth = False
        obs_config.overhead_camera.point_cloud = False
        obs_config.overhead_camera.mask = False

        obs_config.right_shoulder_camera.mask = True
        obs_config.left_shoulder_camera.mask = True
        obs_config.wrist_camera.mask = True
        obs_config.front_camera.mask = True

        obs_config.right_shoulder_camera.depth = True
        obs_config.left_shoulder_camera.depth = True
        obs_config.wrist_camera.depth = True
        obs_config.front_camera.depth = True

        obs_config.right_shoulder_camera.image_size = img_size
        obs_config.left_shoulder_camera.image_size = img_size

        obs_config.wrist_camera.image_size = img_size
        obs_config.front_camera.image_size = img_size

        # Store depth as 0 - 1
        obs_config.right_shoulder_camera.depth_in_meters = False
        obs_config.left_shoulder_camera.depth_in_meters = False
        obs_config.overhead_camera.depth_in_meters = False
        obs_config.wrist_camera.depth_in_meters = False
        obs_config.front_camera.depth_in_meters = False

        # We want to save the masks as rgb encodings.
        # obs_config.left_shoulder_camera.masks_as_one_channel = False
        # obs_config.right_shoulder_camera.masks_as_one_channel = False
        # obs_config.overhead_camera.masks_as_one_channel = False
        # obs_config.wrist_camera.masks_as_one_channel = False
        # obs_config.front_camera.masks_as_one_channel = False

        if config['renderer'] == 'opengl':
            obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
            obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
            obs_config.overhead_camera.render_mode = RenderMode.OPENGL
            obs_config.wrist_camera.render_mode = RenderMode.OPENGL
            obs_config.front_camera.render_mode = RenderMode.OPENGL

        rlbench_env = Environment(
            action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
            obs_config=obs_config,
            headless=True)
        rlbench_env.launch()

        task_env = None
        # endregion
        # Figure out what task/variation this thread is going to do
        task_env = rlbench_env.get_task(task)
        num_all_variations = task_env.variation_count()
        num_each_variation = config['episodes_per_task'] // num_all_variations
        reminder = config['episodes_per_task'] % num_all_variations
        tmp_list = []
        for i in range(num_all_variations):
            tmp_list.append(num_each_variation)
        for i in range(reminder):
            tmp_list[i] += 1

        print('temp_list:', tmp_list)

        for variation_id in range(num_all_variations):

            task_env.set_variation(variation_id)
            descriptions, obs = task_env.reset()

            variation_path = os.path.join(
                config['save_path'], task_env.get_name(),
                VARIATIONS_FOLDER % variation_id
            )
            # print(variation_path)

            check_and_make(variation_path)

            with open(os.path.join(
                    variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                pickle.dump(descriptions, f)

            episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
            check_and_make(episodes_path)

            abort_variation = False
            for episode_id in range(tmp_list[variation_id]):
                # print('Process', i, '// Task:', task_env.get_name(),
                #       '// Variation:', variation_id, '// Demo:', episode_id)
                attempts = 10
                while attempts > 0:
                    episode_path = os.path.join(episodes_path, EPISODE_FOLDER % episode_id)
                    if os.path.exists(episode_path):
                        break
                    try:
                        # TODO: for now we do the explicit looping.
                        demo, = task_env.get_demos(
                            amount=1,
                            live_demos=True)
                    except Exception as e:
                        attempts -= 1
                        if attempts > 0:
                            continue
                        problem = (
                            'Process %d failed collecting task %s (variation: %d, '
                            'example: %d). Skipping this task/variation.\n%s\n' % (
                                i, task_env.get_name(), variation_id, episode_id,
                                str(e))
                        )
                        print(problem)
                        abort_variation = True
                        break

                    post_process_demo(demo, episode_path)
                    pbar.update(1)
                    break
                if abort_variation:
                    break

        rlbench_env.shutdown()


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

seed = 42
config = dict()
config['save_path'] = f'/media/jian/ssd4t/zero/1_Data/A_Selfgen/train/with_sem/'
config['all_task_file'] = '/media/jian/ssd4t/zero/assets/peract_tasks.json'
config['tasks'] = None
config['image_size'] = [512, 512]
config['renderer'] = 'opengl'
config['processes'] = 1
config['episodes_per_task'] = 100
config['variations'] = -1
config['offset'] = 0
config['state'] = False
config['seed'] = seed
# config['tasks'] = ['insert_onto_square_peg']
check_and_make(config['save_path'])

with open(config['all_task_file'], 'r') as f:
    all_tasks = json.load(f)
    if config['tasks']:
        all_tasks = [t for t in all_tasks if t in config['tasks']]

all_tasks = [task_file_to_task_class(t + '_peract') for t in all_tasks]
print('Tasks:', all_tasks)

processes = []
semaphore = Semaphore(config['processes'])


for each_task in all_tasks:
    pbar = tqdm(total=config['episodes_per_task'], desc=f"{each_task.__name__}")
    processes.append(Process(target=run, args=(each_task, semaphore, config, pbar)))
    # break

for p in processes:
    p.start()

for p in processes:
    p.join()

print('All done!')
