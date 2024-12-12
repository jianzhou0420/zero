from torch.utils.data import DataLoader, default_collate
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import yaml

from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from time import time

import torch
from torch.utils.data import Dataset
from zero.v1.dataset.utils import loader, Resize, TrajectoryInterpolator
from datetime import datetime
import pickle
import os
from tqdm import trange
import numpy as np
from typing import Dict, Optional, Sequence
Instructions = Dict[str, Dict[int, torch.Tensor]]


def load_instructions(
    instructions: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None


class RLBenchDataset(Dataset):
    """RLBench dataset."""

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=False,
        refer_list_path=None,

    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length
            )

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                if instructions is not None:
                    self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per task and variation
        episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            episodes_by_task[task] = sorted(
                eps, key=lambda t: int(str(t[2]).split('/')[-1][2:-4])
            )
            self._episodes += eps
            self._num_episodes += len(eps)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_task = episodes_by_task
        print(f"Created dataset from {root} with {self._num_episodes}")
        # 以上东西不管它在做什么，下面是我的dataet的修改代码
        # dataset的思路，全遵循作者的策略，不过不会让每次提取的数据量不一样
        # 首先统计每个episode的长度，制作成list
        # 原始代码里面是随机选取每个episode的一段，虽然总共keyposes是11855个，但每个epoch其实只跑了约7250个，每次不一样，但约等于7250，这里我们直接固定为7250
        #
        # 每次到新的epoch就重新选取一次，

        with open(refer_list_path, 'rb') as f:
            self.refer_list = pickle.load(f)

    def create_refer_list(self, num_epoches, save_path):
        # make sure save_path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        refer_list_episodewise = []
        for episode_id in trange(self._num_episodes):
            task, variation, file = self._episodes[episode_id]

            episode = loader(file)
            if episode is None:
                return None

            chunks = np.array([random.randint(0, math.ceil(len(episode[0]) / self._max_episode_length) - 1) for _ in range(num_epoches)])  # np.random.randint dont accept random from [0,0]
            lower_index = chunks * self._max_episode_length
            upper_index = (chunks + 1) * self._max_episode_length

            frame_ids = np.array(episode[0])

            frame_ids = [frame_ids[lower:upper] for lower, upper in zip(lower_index, upper_index)]  # num_epoches,frame_ids(different length)
            refer_list_episodewise.append(frame_ids)  # num_episodes,num_epoches,frame_ids(different length)
            del episode

        # now i have a refer list, each element in it is a list of num_epoches samples for single episode
        refer_list_epochwise = []
        for epoch_id in trange(num_epoches):
            single_refer_list_epochwise = []
            for episode_id in range(self._num_episodes):
                for frame_id in refer_list_episodewise[episode_id][epoch_id]:
                    single_refer_list_epochwise.append([episode_id, int(frame_id)])
            refer_list_epochwise.append(single_refer_list_epochwise)

        for epoch_id in trange(num_epoches):
            with open(os.path.join(save_path, f'refer_list_epoch_{epoch_id}.pkl'), 'wb') as f:
                pickle.dump(refer_list_epochwise[epoch_id], f)

    def read_from_cache(self, args):
        if self._cache_size == 0:
            return loader(args)

        if args in self._cache:
            return self._cache[args]

        value = loader(args)

        if len(self._cache) == self._cache_size:
            key = list(self._cache.keys())[int(time()) % self._cache_size]
            del self._cache[key]

        if len(self._cache) < self._cache_size:
            self._cache[args] = value

        return value

    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def __len__(self):
        return len(self.refer_list)

    def __getitem__(self, refer_list_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """

        """
        # 作者的dataloader会让每个batch的size变得不一样，这有点不好。为了让其变得一样，我对dataset的代码进行了一点修改
        """
        episode_id, frame_id = self.refer_list[refer_list_id]
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        frame_ids = []
        frame_ids.append(frame_id)
        # print(f"frame_ids: {frame_ids}")

        # print(f"frame_ids: {frame_ids}")
        # Get the image tensors for the frame ids we got
        try:
            states = torch.stack([
                episode[1][i] if isinstance(episode[1][i], torch.Tensor)
                else torch.from_numpy(episode[1][i])
                for i in frame_ids
            ])
        except:
            print("episodeid: ", episode_id, "frame_id: ", frame_id)
            print("episode[0]: ", episode[0])

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0]
        pcds = states[:, :, 1]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        if self._instructions:
            instr = random.choice(self._instructions[task][variation])
            instr = instr[None].repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # gripper history
        gripper_history = torch.stack([
            torch.cat([episode[4][max(0, i - 2)] for i in frame_ids]),
            torch.cat([episode[4][max(0, i - 1)] for i in frame_ids]),
            gripper
        ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [
                    self._interpolate_traj(episode[5][i]) for i in frame_ids
                ]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, 8)
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })

        return ret_dict

    def reset_dataset(self):
        pass


class RLBench3DDADataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage: str):
        instruction = load_instructions(
            self.config['path_instructions'],
            tasks=self.config['tasks'],
            variations=self.config['variations']
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]
        self.epoch_id = 0
        # Initialize datasets with arguments
        self.train_dataset = RLBenchDataset(
            root=self.config['path_dataset'],
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.config['max_episode_length'],
            cache_size=self.config['cache_size'],
            max_episodes_per_task=self.config['max_episodes_per_task'],
            num_iters=self.config['train_iters'],  # 注释掉会让dataset的len变成1800，己
            cameras=self.config['cameras'],
            training=True,
            image_rescale=tuple(
                float(x) for x in self.config['image_rescale'].split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.config['dense_interpolation']),
            interpolation_length=self.config['interpolation_length'],
            refer_list_path=os.path.join(self.config['path_dataset'], 'a_refer_list', f'refer_list_epoch_{self.epoch_id}.pkl'))

    def resetup(self):
        '''
        resetup the dataset with new refer_list
        '''
        instruction = load_instructions(
            self.config['path_instructions'],
            tasks=self.config['tasks'],
            variations=self.config['variations']
        )
        if instruction is None:
            raise NotImplementedError()
        else:
            taskvar = [
                (task, var)
                for task, var_instr in instruction.items()
                for var in var_instr.keys()
            ]
        # Initialize datasets with arguments
        self.train_dataset = RLBenchDataset(
            root=self.config['path_dataset'],
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.config['max_episode_length'],
            cache_size=self.config['cache_size'],
            max_episodes_per_task=self.config['max_episodes_per_task'],
            num_iters=self.config['train_iters'],  # 注释掉会让dataset的len变成1800，己
            cameras=self.config['cameras'],
            training=True,
            image_rescale=tuple(
                float(x) for x in self.config['image_rescale'].split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.config['dense_interpolation']),
            interpolation_length=self.config['interpolation_length'],
            refer_list_path=os.path.join(self.config['path_dataset'], 'a_refer_list', f'refer_list_epoch_{self.epoch_id}.pkl'))

    def train_dataloader(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        g = torch.Generator()
        g.manual_seed(0)
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            worker_init_fn=seed_worker,
            collate_fn=default_collate,
            pin_memory=True,
            # sampler=train_sampler,
            drop_last=True,
            generator=g
        )
        return train_dataloader

    def on_train_epoch_end(self):
        self.epoch_id += 1
        self.resetup()

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


if __name__ == '__main__':
    import json

    def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
        gripper_loc_bounds = json.load(open(path, "r"))
        if task is not None and task in gripper_loc_bounds:
            gripper_loc_bounds = gripper_loc_bounds[task]
            gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
            gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
            gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
        else:
            # Gripper workspace is the union of workspaces for all tasks
            gripper_loc_bounds = json.load(open(path, "r"))
            gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
            gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
            gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
        print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
        return gripper_loc_bounds

    from zero.v1.trainer_3d_da import Trainer3DDA, get_gripper_loc_bounds
    with open('/workspace/zero/zero/v1/config/Diffuser_actor_3d.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['variations'] = tuple(range(200))
    config['gripper_location_boundaries'] = get_gripper_loc_bounds(
        config['path_gripper_location_boundaries'],
        task=config['tasks'][0] if len(config['tasks']) == 1 else None,
        buffer=config['gripper_loc_bounds_buffer'],
    )
    datamodule = RLBench3DDADataModule(config)
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()

    datamodule.train_dataset.create_refer_list(16000, '/media/jian/ssd4t/data/peract/Peract_packaged/train/a_refer_list')
    print(datamodule.train_dataset.__len__())
