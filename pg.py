
# utils
import re
import yaml
import einops
import os
import json
import jsonlines
import torch
import numpy as np
import pickle
from filelock import FileLock
import random
import torch.multiprocessing as mp
from termcolor import colored
from typing import Callable
from copy import deepcopy as copy
from numpy import array as npa
# rlbench
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation

# homemade rlbench
from zero.env.rlbench_lotus.environments import RLBenchEnv, Mover
from zero.env.rlbench_lotus.recorder import TaskRecorder, StaticCameraMotion, CircleCameraMotion, AttachedCameraMotion

# policy & pytorch-lightning


from zero.expForwardKinematics.config.default import build_args
from zero.expForwardKinematics.ObsProcessor.ObsProcessorFKAll import ObsProcessorRLBenchBase
from zero.expForwardKinematics.trainer_FK_all import Trainer_all
# homemade utils
from zero.z_utils.utilities_all import denormalize_JP, denormalize_pos, deconvert_rot


class Actioner(object):
    def __init__(self, eval_config) -> None:
        self.args = eval_config
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)
        self.device = torch.device(eval_config['device'])
        with open(eval_config.model_config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        self.config = config['config']

        model_name = self.config['Trainer']['model_name']
        test = Trainer_all(self.config)
        print('1111111')
        # self.model = model.policy
        # self.model.to(self.device)
        # self.model.eval()

        # self.obs_processor = OBS_FACTORY[model_name](self.config, train_flag=False)  # type: ObsProcessorRLBenchBase
        # self.obs_processor.dataset_init()

        # self.data_container = {
        #     'JP_hist': [],
        #     'eePose_hist': [],
        # }

    def preprocess_obs(self, taskvar, step_id, obs: Observation):
        obs_raw = self.obs_processor.obs_2_obs_raw(obs)
        # 一点中间处理

        JP_curr_no_open = copy(obs_raw['JP_curr_no_open'][0])
        is_open = obs_raw['action'][0][-1]
        JP_curr = np.concatenate((JP_curr_no_open, npa([is_open])), axis=0)
        eePose_curr = copy(obs_raw['action'][0])

        # TODO：迁移到obs_processor里面
        self.update_data_container('JP_hist', JP_curr)
        self.update_data_container('eePose_hist', eePose_curr)
        obs_raw['JP_hist_eval'] = self.data_container['JP_hist']
        obs_raw['eePose_hist_eval'] = self.data_container['eePose_hist']
        obs_static = self.obs_processor.static_process(obs_raw)
        obs_dynamic = self.obs_processor.dynamic_process(obs_static, taskvar)
        batch = self.obs_processor.collect_fn([obs_dynamic])

        for item in batch:
            if isinstance(batch[item], torch.Tensor):
                batch[item] = batch[item].to(self.device)
        return batch

    def predict(self, task_str=None, variation=None, step_id=None, obs_state_dict=None, episode_id=None, instructions=None,):

        # print(obs_state_dict)
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(taskvar, step_id, obs_state_dict,)
        with torch.no_grad():
            if self.config['DiffuserActor']['Policy']['action_space'] == 'JP':
                actions = self.model.inference_one_sample_JP(batch)[0].data.cpu()  # 原本这里是(7) # 现在，这里要变成(horizon_length,7)
            elif self.config['DiffuserActor']['Policy']['action_space'] == 'eePose':
                actions = self.model.inference_one_sample_eePose(batch)[0].data.cpu()
            # actions analysis
            if type(actions) == list:
                actions = torch.stack(actions, 0)
            if len(actions.shape) == 1:
                # single horizon
                actions = actions.unsqueeze(0)
            if len(actions.shape) == 3:
                actions = actions.squeeze(0)
            # check actions shape

        if self.config['DiffuserActor']['Policy']['action_space'] == 'JP':

            actions = denormalize_JP(actions)
        elif self.config['DiffuserActor']['Policy']['action_space'] == 'eePose':
            actions[..., :3] = denormalize_pos(actions[..., :3])
            actions = deconvert_rot(actions)

        new_actions = [npa(actions[i]) for i in range(actions.shape[0])]

        out = {
            'actions': new_actions
        }

        if self.args.save_obs_outs_dir is not None:
            np.save(
                os.path.join(self.args.save_obs_outs_dir, f'{task_str}+{variation}-{episode_id}-{step_id}.npy'),
                {
                    'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                    'obs': obs_state_dict,
                    'action': new_actions
                }
            )
        return out

    def process_and_save_actions(self):
        pass

    def update_data_container(self, name, value):
        length = len(self.data_container[name])
        H = 8  # TODO:horizon

        if length == H:
            self.data_container[name].pop(0)
            self.data_container[name].append(value)
        elif length == 0:
            [self.data_container[name].append(value)for _ in range(H)]
        else:
            raise ValueError(f"data_container {name} length is {length}, but it should be 0 or {H}.")


eval_config = build_args('/data/zero/zero/expForwardKinematics/config/eval_all.yaml')


Actioner1 = Actioner(eval_config)
print('2222222')
