
from tqdm import tqdm
import json
import random
import copy
import os
import pickle
import re
import numpy as np
demo2000_path = "/media/jian/ssd4t/zero/1_Data/A_Selfgen/2000demo_closejar"

vardes_path = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/2000demo_closejar/train/close_jar/variation0/variation_descriptions.pkl'

with open(vardes_path, 'rb') as f:
    vardes = pickle.load(f)

print(vardes)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def find_middle_actions(actions_path, theta_actions_path, sub_keyframe_dection_mode='avg'):
    horizon = 8
    if sub_keyframe_dection_mode == 'avg':
        indices = np.linspace(0, len(actions_path) - 1, horizon + 1).astype(int)[1:]  # 我为什么这里减1了？ 哦index从0开始
        gt_actions = [actions_path[i] for i in indices]
        gt_theta_actions = [theta_actions_path[i] for i in indices]
        return gt_actions, gt_theta_actions
    elif sub_keyframe_dection_mode == 'xyzpeak':
        NotImplementedError("XYZPEAK")


def process_single_episode(root_dir, task, variation, episode):
    out = {
        'rgb': [],
        'pcd': [],
        'action_history': [],
        'action_future': [],
        'txt_embed': [],

    }
    data_folder = os.path.join(root_dir, task, variation, 'episodes', episode)

    with open(os.path.join(data_folder, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(data_folder, 'actions_all.pkl'), 'rb') as f:
        actions_all = pickle.load(f)
    with open(os.path.join(data_folder, 'positions_all.pkl'), 'rb') as f:
        positions_all = pickle.load(f)
    save_root = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DA3D'
    save_folder = os.path.join(save_root, task, variation, 'episodes', episode)
    num_frames = len(data['rgb']) - 1

    taskvar_instrs = json.load(open('/data/zero/assets/taskvars_instructions_peract.json'))
    instr_embeds = np.load('/data/zero/assets/instr_embeds_clip.npy', allow_pickle=True).item()

    taskvar = task + '_' + 'peract+' + variation.split('variation')[1]

    for i in range(num_frames):
        keyframe_id = copy.deepcopy(np.array(data['key_frameids'][i], dtype=np.int16))
        rgb = data['rgb'][i]
        pcd = data['pc'][i]

        action_current = copy.deepcopy(np.array(data['action'][i], dtype=np.float16))
        action_next = copy.deepcopy(np.array(data['action'][i + 1], dtype=np.float16))
        actions_path = copy.deepcopy(np.array(actions_all[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float16))
        theta_actions_path = copy.deepcopy(np.array(positions_all[data['key_frameids'][i]:data['key_frameids'][i + 1] + 1], dtype=np.float16))
        assert (action_current - actions_all[keyframe_id] < 0.001).all()
        if keyframe_id == 0:
            action_history = [action_current] * 8
        elif keyframe_id - 8 <= 1:
            action_history = [actions_all[j] for j in range(keyframe_id)]
            action_history += [action_current] * (8 - keyframe_id)
        else:
            action_history = [actions_all[j] for j in range(keyframe_id - 8, keyframe_id + 1)]

        action_future, theta_action_future = find_middle_actions(actions_path, theta_actions_path, sub_keyframe_dection_mode='avg')

        assert (action_next - action_future[-1] < 0.001).all()
        assert (action_current - action_history[-1] < 0.001).all()

        action_future = actions_all

        instr = random.choice(taskvar_instrs[taskvar])
        instr_embed = copy.deepcopy(instr_embeds[instr])

        out['rgb'].append(rgb)
        out['pcd'].append(pcd)
        out['action_history'].append(action_history)
        out['action_future'].append(action_future)
        out['txt_embed'].append(instr_embed)

    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, 'data.pkl'), 'wb') as f:
        pickle.dump(out, f)


data_dir = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/2000demo_closejar/train'
tasks_all = sorted(os.listdir(data_dir), key=natural_sort_key)

for i, task in enumerate(tasks_all):
    variations = sorted(os.listdir(os.path.join(data_dir, task)), key=natural_sort_key)
    for j, variation in enumerate(variations):
        episodes = sorted(os.listdir(os.path.join(data_dir, task, variation, 'episodes')), key=natural_sort_key)
        for k, episode in tqdm(enumerate(episodes)):
            process_single_episode(data_dir, task, variation, episode)
            # print(task, variation, episode)
            # print(i, j, k)
