import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import json
from datasets import load_dataset


class JianAlohaDataset(Dataset):
    '''
    load hdf5 files from the specified directory
    '''

    def __init__(self, data_dir='/data/aloha_sim_insertion_human'):
        self.data_dir = data_dir
        self.info_json = json.load(open(os.path.join(data_dir, 'meta', 'info.json'), 'r'))
        self.stats_json = json.load(open(os.path.join(data_dir, 'meta', 'stats.json'), 'r'))

        self.episodes_jsonl = []
        with open(os.path.join(data_dir, 'meta', 'episodes.jsonl'), 'r') as f:
            for line in f:
                self.episodes_jsonl.append(json.loads(line))

        self.tasks_jsonl = []
        with open(os.path.join(data_dir, 'meta', 'tasks.jsonl'), 'r') as f:
            for line in f:
                self.tasks_jsonl.append(json.loads(line))

    def __len__(self):
        return self.info_json['total_frames']

    def __getitem__(self, idx):
        '''
        assume idx is the frame number in the dataset
        '''
        episode_id, frame_id = self._identify_episode(idx)
        data_path = os.path.join(self.data_dir, 'data/chunk-000/', f'episode_{episode_id:06d}.parquet')
        video_path = os.path.join(self.data_dir, 'videos/chunk-000/', f'episode_{episode_id:06d}.mp4')
        df = pd.read_parquet(data_path)
        return df.iloc[frame_id]

    def _identify_episode(self, idx):
        '''
        identify the episode number and the frame number in the episode
        '''
        episode_num = 0
        frame_num = 0
        for i in range(len(self.episodes_jsonl)):
            if idx < self.episodes_jsonl[i]['length']:
                episode_num = i
                frame_num = idx
                break
            else:
                idx -= self.episodes_jsonl[i]['length']

        return episode_num, frame_num


if __name__ == '__main__':
    test = JianAlohaDataset()
    print(test[0])
    print(len(test))
