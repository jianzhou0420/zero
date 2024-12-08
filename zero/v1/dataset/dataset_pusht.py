import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import json
import cv2
from PIL import Image
import torch


class JianPushTDataset(Dataset):
    '''
    load hdf5 files from the specified directory
    '''

    def __init__(self, data_dir='/data/pusht/'):
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
        return_dict = dict()
        episode_id, frame_id = self._identify_episode(idx)
        data_path = os.path.join(self.data_dir, 'data/chunk-000/', f'episode_{episode_id:06d}.parquet')
        df_data = pd.read_parquet(data_path)
        this_frame_data = df_data.iloc[frame_id]
        for key, value in this_frame_data.items():
            return_dict[key] = torch.tensor(value, dtype=torch.float32)

        # load the image
        image_path = os.path.join(self.data_dir, f'videos/chunk-000/observation.image/episode_{episode_id:06d}/frame_{frame_id:04d}.png')
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).reshape(3, 96, 96)
        image = image / 255.0
        return_dict['image'] = torch.tensor(image, dtype=torch.float32)

        return return_dict

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
    test = JianPushTDataset()
    print(test[0].keys())
    print(test.__len__())
