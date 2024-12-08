# framework package
from torch.utils.data import DataLoader, default_collate
import random
import torch.optim as optim
import pickle
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F

# My core package
from zero.v1.dataset.datasets_18_tasks import JianRLBenchDataset
from zero.v1.models.zero_test import ZeroModel
from zero.v1.dataset.dataset_pusht import JianPushTDataset
from typing import Dict, Optional, Sequence
# utils package
import yaml
from datetime import datetime
import argparse
from pathlib import Path
from zero.v1.dataset.dataset_3dda_rlbench import RLBenchDataset
from zero.v1.models.diffuser_actor import DiffuserActor
import json
Instructions = Dict[str, Dict[int, torch.Tensor]]
'''
Jian: To make my code clean, this file only contain the code of trainning and evaluation.
      Details of the model can be found in CVAE.py
'''
torch.set_float32_matmul_precision('medium')


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


class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, mask=None, is_loss=True):
        if not is_loss:
            assert gt is not None and mask is not None
            return self.compute_metrics(pred, gt, mask)[0]['action_mse']
        return pred

    @staticmethod
    def compute_metrics(pred, gt, mask):
        # pred/gt are (B, L, 7), mask (B, L)
        pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
        # symmetric quaternion eval
        quat_l1 = (pred[..., 3:7] - gt[..., 3:7]).abs().sum(-1)
        quat_l1_ = (pred[..., 3:7] + gt[..., 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        # gripper openess
        openess = ((pred[..., 7:] >= 0.5) == (gt[..., 7:] > 0.0)).bool()
        tr = 'traj_'

        # Trajectory metrics
        ret_1, ret_2 = {
            tr + 'action_mse': F.mse_loss(pred, gt),
            tr + 'pos_l2': pos_l2.mean(),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
            tr + 'rot_l1': quat_l1.mean(),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
            tr + 'gripper': openess.flatten().float().mean()
        }, {
            tr + 'pos_l2': pos_l2.mean(-1),
            tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
            tr + 'rot_l1': quat_l1.mean(-1),
            tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
        }

        # Keypose metrics
        pos_l2 = ((pred[:, -1, :3] - gt[:, -1, :3]) ** 2).sum(-1).sqrt()
        quat_l1 = (pred[:, -1, 3:7] - gt[:, -1, 3:7]).abs().sum(-1)
        quat_l1_ = (pred[:, -1, 3:7] + gt[:, -1, 3:7]).abs().sum(-1)
        select_mask = (quat_l1 < quat_l1_).float()
        quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
        ret_1.update({
            'pos_l2_final': pos_l2.mean(),
            'pos_l2_final<0.01': (pos_l2 < 0.01).float().mean(),
            'rot_l1': quat_l1.mean(),
            'rot_l1<0025': (quat_l1 < 0.025).float().mean()
        })
        ret_2.update({
            'pos_l2_final': pos_l2,
            'pos_l2_final<0.01': (pos_l2 < 0.01).float(),
            'rot_l1': quat_l1,
            'rot_l1<0.025': (quat_l1 < 0.025).float(),
        })

        return ret_1, ret_2


class Trainer3DDA(pl.LightningModule):
    def __init__(self, config_first_layer):
        super().__init__()
        self.config = config_first_layer
        self.model = DiffuserActor(
            backbone=self.config['backbone'],
            image_size=tuple(int(x) for x in self.config['image_size'].split(",")),
            embedding_dim=self.config['embedding_dim'],
            num_vis_ins_attn_layers=self.config['num_vis_ins_attn_layers'],
            use_instruction=bool(self.config['use_instruction']),
            fps_subsampling_factor=self.config['fps_subsampling_factor'],
            gripper_loc_bounds=self.config['gripper_location_boundaries'],
            rotation_parametrization=self.config['rotation_parametrization'],
            quaternion_format=self.config['quaternion_format'],
            diffusion_timesteps=self.config['diffusion_timesteps'],
            nhist=self.config['num_history'],
            relative=bool(self.config['relative_action']),
            lang_enhanced=bool(self.config['lang_enhanced'])
        )
        self.criterion = TrajectoryCriterion()

        self.path = dict()

    def configure_optimizers(self):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.config['lr']},
            {"params": [], "weight_decay": 5e-4, "lr": self.config['lr']}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    def training_step(self, data_dict, idx):

        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].squeeze().cuda()
        """Run a single training step."""

        if self.config['keypose_only']:
            data_dict["trajectory"] = data_dict["trajectory"][:, [-1]]
            data_dict["trajectory_mask"] = data_dict["trajectory_mask"][:, [-1]]
        else:
            data_dict["trajectory"] = data_dict["trajectory"][:, 1:]
            data_dict["trajectory_mask"] = data_dict["trajectory_mask"][:, 1:]

        # Forward pass
        curr_gripper = (
            data_dict["curr_gripper"] if self.config['num_history'] < 1
            else data_dict["curr_gripper_history"][:, -self.config['num_history']:]
        )
        out = self.model(
            data_dict["trajectory"],
            data_dict["trajectory_mask"],
            data_dict["rgbs"],
            data_dict["pcds"],
            data_dict["instr"],
            curr_gripper
        )
        loss = self.criterion.compute_loss(out)

        return loss

    #######################################
    # internal methods
    #######################################

    def _forward_pass_rlbench(self, data_dict):
        '''
        receive data and return the loss
        '''
        if data_dict['images'].shape[-1] == 3:
            data_dict['images'] = data_dict['images'].cuda().permute(0, 1, 4, 2, 3)
        model_output = self.model(data_dict['text'], data_dict['images'])
        return model_output

    def _forward_pass_pusht(self, data_dict):
        '''
        receive data and return the loss
        '''
        text = 'push T to the right place'
        text = [text for _ in range(data_dict['image'].shape[0])]
        images = data_dict['image'].cuda()
        images = images.unsqueeze(1)
        model_output = self.model(text, images)

        return model_output

    #######################################
    # interface methods
    #######################################
    def get_dataset(self):
        """Initialize datasets."""
        """
        datasets already batched, so no need to use DataLoader
        """
        # Load instruction, based on which we load tasks/variations
        tasks = self.config['tasks']
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
        train_dataset = RLBenchDataset(
            root=self.config['path_dataset'],
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.config['max_episode_length'],
            cache_size=self.config['cache_size'],
            max_episodes_per_task=self.config['max_episodes_per_task'],
            num_iters=self.config['train_iters'],
            cameras=self.config['cameras'],
            training=True,
            image_rescale=tuple(
                float(x) for x in self.config['image_rescale'].split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.config['dense_interpolation']),
            interpolation_length=self.config['interpolation_length']
        )
        test_dataset = RLBenchDataset(
            root=self.config['path_valset'],
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=self.config['max_episode_length'],
            cache_size=self.config['cache_size_val'],
            max_episodes_per_task=self.config['max_episodes_per_task'],
            cameras=self.config['cameras'],
            training=False,
            image_rescale=tuple(
                float(x) for x in self.config['image_rescale'].split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.config['dense_interpolation']),
            interpolation_length=self.config['interpolation_length']
        )
        return train_dataset, test_dataset

    def get_dataloader(self):
        train_dataset, test_dataset = self.get_dataset()

        bs = self.config['batch_size']
        shuffle = self.config['shuffle']
        num_workers = self.config['num_workers']

        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        test_dataloader = None

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        g = torch.Generator()
        g.manual_seed(0)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
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
        return train_dataloader, test_dataloader


if __name__ == '__main__':
    with open('/workspace/zero/zero/v1/config/Diffuser_actor_3d.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['variations'] = tuple(range(200))
    config['gripper_location_boundaries'] = get_gripper_loc_bounds(
        config['path_gripper_location_boundaries'],
        task=config['tasks'][0] if len(config['tasks']) == 1 else None,
        buffer=config['gripper_loc_bounds_buffer'],
    )
    trainer_pl = Trainer3DDA(config)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dataloader, test_dataloader = trainer_pl.get_dataloader()
    dataset_name = config['dataset_name']

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_last=True,
        filename=f'{current_time}' + f'{dataset_name}' + '{epoch:03d}'  # Checkpoint filename
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         max_epochs=1,
                         devices=1,
                         strategy='auto',
                         default_root_dir='/data/ckpt',
                         )

    trainer.fit(trainer_pl, train_dataloader)
