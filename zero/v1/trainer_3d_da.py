# framework package
import os
from pytorch_lightning.strategies import DDPStrategy
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
from zero.v1.dataset.dataset_3dda_rlbench_7250 import RLBench3DDADataModule
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
        # print(data_dict['rgbs'])

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

    #######################################
    # interface methods
    #######################################


if __name__ == '__main__':

    '''
    Dataset should in the following order:
    /pathtodata/peract/Peract_packaged/train"
    /pathtodata/peract/Peract_packaged/val"
    /pathtodata/peract/Peract_packaged/instruction.pkl"
    /pathtodata/peract/Peract_packaged/train/a_refer_list
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--homepath', type=str, default='/hpcfs/users/a1946536/zero/zero/v1/')
    parser.add_argument('--datapath', type=str, default='/ssd/')
    args = parser.parse_args()
    config_path = os.path.join(args.homepath, 'config/Diffuser_actor_3d.yaml')
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config['variations'] = tuple(range(200))

    config['path_train_dataset'] = os.path.join(args.datapath, config['relative_path_train_dataset'])
    config['path_val_dataset'] = os.path.join(args.datapath, config['relative_path_val_dataset'])
    config['path_instructions'] = os.path.join(args.datapath, config['relative_path_instructions'])
    path_gripper_location_boundaries = os.path.join(args.homepath, config['relative_path_gripper_location_boundaries'])

    config['gripper_location_boundaries'] = get_gripper_loc_bounds(
        path_gripper_location_boundaries,
        task=config['tasks'][0] if len(config['tasks']) == 1 else None,
        buffer=config['gripper_loc_bounds_buffer'],
    )

    from pytorch_lightning import seed_everything
    seed_everything(42)

    trainer_pl = Trainer3DDA(config)
    datamodule = RLBench3DDADataModule(config)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name = config['dataset_name']

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=200,
        save_last=True,
        filename=f'{current_time}' + f'{dataset_name}' + '{epoch:03d}'  # Checkpoint filename
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         max_epochs=16000,
                         devices='auto',
                         #  strategy="auto",
                         strategy=DDPStrategy(find_unused_parameters=True),
                         default_root_dir='/hpcfs/users/a1946536/log/',
                         reload_dataloaders_every_n_epochs=1,  # help mimic the behavior of 3dda
                         )

    trainer.fit(trainer_pl, datamodule=datamodule)
