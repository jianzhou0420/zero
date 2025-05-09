
from tqdm import tqdm
import os
import pickle
from zero.z_utils.coding import natural_sort_key
import pandas as pd


data_dir = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DP_traj/trajectory/test2/42/train'  # 因为namesapce不高亮，所以尽量用字典的方式，方便区分

# tasks_to_use
tasks_to_use = None
tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)
tasks_all = [t for t in tasks_all if t in tasks_to_use] if tasks_to_use is not None else tasks_all
# 1. episodes-wise list
g_episode_to_path = []  # retrieve all episodes path and put them in episodes
frames = []  # How many frames in each episode
for task_name in tasks_all:
    task_folder_path = os.path.join(data_dir, task_name)
    variation_list = sorted(os.listdir(task_folder_path), key=natural_sort_key)
    for variation_folder in variation_list:
        variation_folder_path = os.path.join(task_folder_path, variation_folder)
        if len(os.listdir(variation_folder_path)) <= 1:
            variation_folder_path = os.path.join(task_folder_path, variation_folder, 'episodes')
        episodes_list = sorted(os.listdir(variation_folder_path), key=natural_sort_key)
        for episode_folder in episodes_list:
            episode_folder_path = os.path.join(variation_folder_path, episode_folder)
            g_episode_to_path.append(episode_folder_path)


action_list = []
pbar = tqdm(total=len(g_episode_to_path), desc="Loading data")
for episode_folder_path in g_episode_to_path:
    with open(os.path.join(episode_folder_path, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
        action_list.extend([data['eePose'][i], data['JP'][i]] for i in range(len(data['eePose'])))
        pbar.update(1)


with open('data.pkl', 'wb') as f:
    pickle.dump(action_list, f)
