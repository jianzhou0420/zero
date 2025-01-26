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
        relative_action=False
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

        # 以上东西不管它在做什么，下面是我的dataet的修改代码
        # dataset的思路，全遵循作者的策略，不过不会让每次提取的数据量不一样
        # 首先统计每个episode的长度，制作成list
        # 原始代码里面是随机选取每个episode的一段，虽然总共keyposes是11855个，但每个epoch其实只跑了约7250个，每次不一样，但约等于7250，这里我们直接固定为7250
        #

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
        if self._num_iters is not None:
            return self._num_iters

        return self._num_episodes

    def __getitem__(self, episode_id):
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
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # 以下复杂公式是为了，随机从episode里面选取在max_episode_length范围内的任意一段frame_ids，且尽可能=max_episode_length
        # 这有用么，反正它最后都是
        # Get frame ids for this chunk
        frame_ids = episode[0][chunk * self._max_episode_length:
                               (chunk + 1) * self._max_episode_length
                               ]
        # print(f"frame_ids: {frame_ids}")

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

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


if __name__ == '__main__':
    from zero.v1.trainer_3d_da import Trainer3DDA, get_gripper_loc_bounds
    with open('/data/zero/zero/v1/config/Diffuser_actor_3d.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['variations'] = tuple(range(200))
    config['gripper_location_boundaries'] = get_gripper_loc_bounds(
        config['path_gripper_location_boundaries'],
        task=config['tasks'][0] if len(config['tasks']) == 1 else None,
        buffer=config['gripper_loc_bounds_buffer'],
    )
    trainer_pl = Trainer3DDA(config)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dataset, test_dataset = trainer_pl.get_dataset()

    for i in range(100):
        print(len(train_dataset))

    # 一次600,000 个,所以，只需要1个epoch就是作者说的
