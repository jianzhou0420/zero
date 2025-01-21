import pytorch_lightning as pl
import torch
import yacs.config
from zero.v1.trainer_lotus import TrainerLotus

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from zero.v1.dataset.datasets_18_tasks import JianRLBenchDataset, normalize_image, normalize_position, denormalize_position
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import MT15_V1, CloseJar
from rlbench.observation_config import ObservationConfig
import yaml

from rlbench.tasks import CloseJar, InsertOntoSquarePeg, LightBulbIn, MeatOffGrill, OpenDrawer, \
    PlaceCups, PlaceShapeInShapeSorter, PushButtons, PutGroceriesInCupboard, \
    PutItemInDrawer, PutMoneyInSafe, ReachAndDrag, StackBlocks, \
    StackCups, TurnTap, SlideBlockToColorTarget, SweepToDustpanOfSize, PlaceWineAtRackLocation

from rlbench.backend.task import Task
'''
18 tasks from peract
'''
tasks_list = [OpenDrawer,
              SlideBlockToColorTarget,
              SweepToDustpanOfSize,
              MeatOffGrill,
              TurnTap,
              PutItemInDrawer,
              CloseJar,
              ReachAndDrag,
              StackBlocks,
              LightBulbIn,
              PutMoneyInSafe,
              PlaceWineAtRackLocation,
              PutGroceriesInCupboard,
              PlaceShapeInShapeSorter,
              PushButtons,
              InsertOntoSquarePeg,
              StackCups,
              PlaceCups]


class EvalGeneral():
    def __init__(self, model, preprocess_fn, postprocess_fn):
        self.mdoel = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    @torch.no_grad()
    def evaluate_tasks_all(self):
        # 1. Set parameters
        # 1.1 same parameters of evaluation
        eval_episodes = 2
        max_time_steps = 180
        self.model.eval()

        # 1.2 functions that are only used here
        # 2. Set up simulators
        # 2.1 Environment setup
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(absolute_mode=True),
            gripper_action_mode=Discrete()
        )

        observation_config = ObservationConfig()
        observation_config.gripper_joint_positions = True

        env = Environment(action_mode, obs_config=observation_config, headless=False)
        env.launch()

        # 3. main loop
        for task_idx in range(len(tasks_list)):
            self.evaluate_specific_tasks(env, tasks_list[task_idx], eval_episodes, max_time_steps)

    def evaluate_specific_tasks(self, env=None, task: Task = CloseJar, eval_episodes=1, max_time_steps=180):
        if env is None:
            self.model.eval()
            # 1.2 functions that are only used here
            # 2. Set up simulators
            # 2.1 Environment setup
            action_mode = MoveArmThenGripper(
                arm_action_mode=JointPosition(absolute_mode=True),
                gripper_action_mode=Discrete()
            )

            observation_config = ObservationConfig()
            observation_config.gripper_joint_positions = True

            env = Environment(action_mode, obs_config=observation_config, headless=False)
            env.launch()

        # task: a rlbench task object
        task = env.get_task(task)
        task.sample_variation()  # random variation
        des, obs = task.reset()
        text = str(des[np.random.randint(len(des))])
        terminate = False
        for episode in range(eval_episodes):
            for t in range(max_time_steps):
                _, images = self.process_obs(obs)  # images: [1, 5, 3, 128, 128]
                data_dict = {'text': text, 'images': images}
                predicted_action_normalized = self.model._forward_pass(data_dict)
                predicted_action_normalized = predicted_action_normalized.squeeze().detach().cpu().numpy()
                predicted_action_normalized[:7] = denormalize_position(predicted_action_normalized[:7])
                real_world_action = predicted_action_normalized

                obs, reward, terminate = task.step(real_world_action)
                terminate = terminate.T
            success, _ = task._task.success()
            print(f'Episode {episode} success: {success}')
            task.sample_variation()  # random variation
            des, obs = task.reset()
            text = str(des[np.random.randint(len(des))])


if __name__ == '__main__':
    config = yacs.config.CfgNode(new_allowed=True)
    config.merge_from_file('/data/zero/zero/v1/config/lotus.yaml')
    ckpt_path = '/data/zero/zero/v1/lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt'
    model = TrainerLotus.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)
    eval_general = EvalGeneral(model)
    eval_general.evaluate_tasks_all()
