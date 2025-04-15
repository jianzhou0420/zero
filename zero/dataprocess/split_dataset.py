import os
import sys
import re
import pickle
import numpy as np
import shutil

train_root = '/data/zero/1_Data/B_Preprocess/FK/1000_train_eval/train/'
tasks = os.listdir(train_root)
counter = 0
episode_list = []
for task_s in tasks:
    task_s_path = os.path.join(train_root, task_s)
    variations = os.listdir(task_s_path)
    for variation in variations:
        variation_path = os.path.join(task_s_path, variation)
        episodes = os.listdir(variation_path)
        for episode in episodes:
            episode_path = os.path.join(variation_path, episode)
            data_file = os.path.join(episode_path, 'data.pkl')
            episode_list.append(data_file)
            if not os.path.exists(data_file):

                print(f"File not found: {data_file}")
            else:
                counter += 1
                print('counter:', counter)

chosen = np.random.choice(episode_list, 100, replace=False)

for src in chosen:
    destination = src.replace('/train/', '/eval/')
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    shutil.move(src, destination)
    src_dir = os.path.dirname(src)
    os.rmdir(src_dir)
