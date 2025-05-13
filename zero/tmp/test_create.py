
import copy
from zero.z_utils.coding import natural_sort_key
import os
from zero.tmp.replay_buffer import ReplayBuffer
import pickle

import numpy as np
data_path = "./1_Data/A_Selfgen/trajectory/test2/42"
zarr_path = './my_replay_buffer.zarr'

replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')


def process_data(obs_raw):
    # Process the data as needed
    # For example, you can extract specific information or transform it
    rgb = obs_raw['rgb']
    eePose = np.array(obs_raw['eePose_all'])  # 3+4+1
    JP = np.array(obs_raw['JP_all'])  # 7+1

    assert len(rgb) == len(eePose) == len(JP), "Data lengths do not match"
    assert isinstance(rgb, np.ndarray), "RGB data is not a list"
    assert isinstance(eePose, np.ndarray), "EE Pose data is not a list"
    assert isinstance(JP, np.ndarray), "JP data is not a list"
    return rgb, eePose, JP


tasks_list = sorted(os.listdir(data_path), key=natural_sort_key)
for i, task in enumerate(tasks_list):
    this_task_path = os.path.join(data_path, task)
    variations_list = sorted(os.listdir(this_task_path), key=natural_sort_key)
    for j, variation in enumerate(variations_list):
        this_variation_path = os.path.join(this_task_path, variation, 'episodes')
        episodes_list = sorted(os.listdir(this_variation_path), key=natural_sort_key)
        for k, episode in enumerate(episodes_list):
            this_episode_path = os.path.join(this_variation_path, episode)
            with open(os.path.join(this_episode_path, 'data.pkl'), 'rb') as f:
                obs_raw = pickle.load(f)

                rgb, eePose, JP = process_data(obs_raw)
                data_dict = {
                    'rgb': rgb,
                    'eePose': eePose,
                    'JP': JP
                }
                # Add the episode data to the replay buffer
                replay_buffer.add_episode(data_dict)
                del data_dict

pass
