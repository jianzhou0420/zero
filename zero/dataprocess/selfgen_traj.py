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
from zero.expForwardKinematics.ObsProcessor.ObsProcessorBase import ObsProcessorRLBenchBase

import numpy as np

from zero.expForwardKinematics.config.default import get_config
import json
from scipy.spatial.transform import Rotation as R

from zero.dataprocess.utils import convert_gripper_pose_world_to_image, keypoint_discovery


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class DataGenerator:
    def __init__(self, config):
        self.data_gen_config = config
        yaml_config = None
        self.obs_processor = ObsProcessorRLBenchBase(yaml_config)

        self.traj_flag = self.data_gen_config['traj_flag']

    def _get_obs_config(self):
        img_size = list(map(int, self.data_gen_config['image_size']))
        obs_config = ObservationConfig()
        obs_config.set_all(True)

        obs_config.overhead_camera.rgb = True
        obs_config.overhead_camera.depth = True
        obs_config.overhead_camera.point_cloud = True
        obs_config.overhead_camera.mask = True

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
        obs_config.overhead_camera.image_size = img_size
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

        if self.data_gen_config['renderer'] == 'opengl':
            obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
            obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
            obs_config.overhead_camera.render_mode = RenderMode.OPENGL
            obs_config.wrist_camera.render_mode = RenderMode.OPENGL
            obs_config.front_camera.render_mode = RenderMode.OPENGL

        return obs_config

    def generate_single_variation_by_num(self, task, var, num, config, pbar):
        """Each thread will choose one task and variation, and then gather
        all the episodes_per_task for that variation."""

        # Initialise each thread with random seed
        # np.random.seed(None)
        np.random.seed(config['seed'])
        random.seed(config['seed'])

        obs_config = self._get_obs_config()
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
        if var >= num_all_variations:
            print('Variation %d does not exist for task %s' % (var, task_env.get_name()))
            return

        # print('temp_list:', tmp_list)

        task_env.set_variation(var)
        descriptions, obs = task_env.reset()

        variation_path = os.path.join(
            config['save_path'], task_env.get_name(),
            VARIATIONS_FOLDER % var
        )

        check_and_make(variation_path)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for episode_id in range(num):
            # print('Process', i, '// Task:', task_env.get_name(),
            #       '// Variation:', variation_id, '// Demo:', episode_id)
            attempts = 1
            while attempts > 0:
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % episode_id)
                if os.path.exists(episode_path):
                    break
                try:
                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                    # with open('./5_templates/demo.pkl', 'wb') as f:
                    #     pickle.dump(demo, f)
                    self.demo2data(demo, episode_path)
                    pbar.update(1)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = ('error')
                    print(problem)
                    abort_variation = True
                    break

                # self.demo2data(demo, episode_path)
                break
            if abort_variation:
                break

        rlbench_env.shutdown()

    def demo2data(self, demo, example_path):
        '''
        interface between DataGenerator and Ptv3Perceptor
        '''
        out = self.obs_processor.demo_2_obs_raw(demo, self.traj_flag)
        check_and_make(example_path)
        with open(os.path.join(example_path, 'data.pkl'), 'wb') as f:
            pickle.dump(out, f)

    @staticmethod
    def generate_data_single_process():
        seed = 42
        config = get_config('./zero/expForwardKinematics/config/datagen.yaml')
        config.defrost()
        config['seed'] = seed
        config['save_path'] = os.path.join(config['save_path'], str(seed))
        check_and_make(config['save_path'])

        all_tasks = config['task']
        all_tasks = [task_file_to_task_class(t) for t in all_tasks]
        print('Tasks:', all_tasks)
        pbar = tqdm(total=len(config['var']) * config['num'], desc=f"{all_tasks[0].__name__}")
        for i, each_task in enumerate(all_tasks):
            test = DataGenerator(config)
            for j, var in enumerate(config['var']):
                test.generate_single_variation_by_num(each_task, var, config['num'], config, pbar)
            # break


if __name__ == '__main__':
    DataGenerator.generate_data_single_process()
