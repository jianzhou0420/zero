

from zero.expForwardKinematics.ObsProcessor.ObsProcessorPtv3 import ObsProcessorPtv3
from zero.expForwardKinematics.config.default import get_config
import re
import os


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def obs_raw_to_static_process():
    config_path = "/data/zero/zero/expForwardKinematics/config/DP.yaml"
    config = get_config(config_path)
    obs_processor = ObsProcessorPtv3(config, train_flag=True)
    obs_raw_path = "/data/zero/1_Data/A_Selfgen/2000demo_put_groceries/train/904744"
    tasks_list = sorted(os.listdir(obs_raw_path), key=natural_sort_key)

    for i, task in enumerate(tasks_list):
        this_task_path = os.path.join(obs_raw_path, task)
        variations_list = sorted(os.listdir(this_task_path), key=natural_sort_key)
        for j, variation in enumerate(variations_list):
            this_variation_path = os.path.join(this_task_path, variation, 'episodes')
            episodes_list = sorted(os.listdir(this_variation_path), key=natural_sort_key)
            for k, episode in enumerate(episodes_list):
                this_episode_path = os.path.join(this_variation_path, episode)


obs_raw_to_static_process()
