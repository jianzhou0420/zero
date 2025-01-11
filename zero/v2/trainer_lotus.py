# framework package

from zero.v1.models.lotus.optim.misc import build_optimizer
from zero.v1.dataset.dataset_lotus_voxelexp_copy import SimplePolicyDataset, ptv3_collate_fn
from zero.v1.models.lotus.simple_policy_ptv3 import SimplePolicyPTV3CA
import argparse
from datetime import datetime
import yaml
import yacs.config
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from typing import List, Dict, Tuple, Union, Iterator
from argparse import Namespace
import math
import os
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
import warnings

warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")

# utils package
#
torch.set_float32_matmul_precision('medium')


class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return current_step / warmup_steps
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            return max(min_lr, cosine_decay)

        super().__init__(optimizer, lr_lambda)


class TrainerLotus(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SimplePolicyPTV3CA(config.MODEL)

    def training_step(self, batch, batch_idx):  # 每次的batch_size都是不一样的应该说，每个小batch的每一个sample，sample的长度是不一样的

        losses = self.model(batch, is_train=True)
        self.log('train_loss', losses['total'])
        # if self.global_step % 10 == 0:
        #     print(f"train_loss: {losses['total']}")
        # print(f"After loading: {torch.cuda.memory_allocated()} bytes")
        return losses['total']

    def configure_optimizers(self):
        optimizer, init_lrs = build_optimizer(self.model, self.config.TRAIN)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=self.config.TRAIN.warmup_steps,
            total_steps=self.config.TRAIN.num_train_steps,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Adjust learning rate every step
        }
        return [optimizer], [scheduler_config]

    def get_dataset(self, config):
        dataset = SimplePolicyDataset(**config.TRAIN_DATASET)
        return dataset

    def get_dataloader(self, config):
        def build_dataloader(dataset, collate_fn, is_train: bool, config):
            '''
            copied from lotus, sampler is not used as pytorchlightning will automaticallt config it for me
            '''
            batch_size = config.TRAIN.train_batch_size if is_train else config.TRAIN.val_batch_size
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=config.TRAIN.n_workers,
                pin_memory=config.TRAIN.pin_mem,
                collate_fn=collate_fn,
                drop_last=False,
                prefetch_factor=2 if config.TRAIN.n_workers > 0 else None,
            )
            return loader
        # function
        dataset = self.get_dataset(config)
        train_loader = build_dataloader(dataset, ptv3_collate_fn, is_train=True, config=config)
        return train_loader


if __name__ == '__main__':
    def train(voxel_size):
        config = yacs.config.CfgNode(new_allowed=True)
        config.merge_from_file(f'/workspace/zero/zero/v1/config/lotus.yaml')

        config.TRAIN_DATASET.tasks_to_use = ['close_jar']
        trainer_model = TrainerLotus(config)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_dataloader = trainer_model.get_dataloader(config)

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=500,
            save_top_k=-1,
            save_last=False,
            filename=f'{current_time}' + '{epoch:03d}'  # Checkpoint filename
        )
        csvlogger1 = CSVLogger('/data/logs', name=f'voxel{voxel_size}')

        max_epochs = int(1500)
        print(f"config.TRAIN.num_train_steps: {config.TRAIN.num_train_steps}")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"max_epochs: {max_epochs}")
        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=max_epochs,
                             devices='auto',
                             strategy='auto',
                             logger=csvlogger1,)

        trainer.fit(trainer_model, train_dataloader)

    # def train_resume(ckpt_path):
    #     config = yacs.config.CfgNode(new_allowed=True)
    #     config.merge_from_file(f'/workspace/zero/zero/v1/config/lotus.yaml')

    #     config.TRAIN_DATASET.tasks_to_use = ['close_jar']
    #     trainer_model = TrainerLotus(config)
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     train_dataloader = trainer_model.get_dataloader(config)
    #     # ckpt_path = '/media/jian/ssd4t/logs/voxelNone/version_4/checkpoints/20250111_162332epoch=1499.ckpt'
    #     checkpoint_callback = ModelCheckpoint(
    #         every_n_epochs=500,
    #         save_top_k=-1,
    #         save_last=False,
    #         filename=f'{current_time}' + '{epoch:03d}'  # Checkpoint filename
    #     )
    #     csvlogger1 = CSVLogger('/data/logs', name=f'voxel{None}')

    #     max_epochs = int(6500)
    #     print(f"config.TRAIN.num_train_steps: {config.TRAIN.num_train_steps}")
    #     print(f"len(train_dataloader): {len(train_dataloader)}")
    #     print(f"max_epochs: {max_epochs}")
    #     trainer = pl.Trainer(callbacks=[checkpoint_callback],
    #                          max_epochs=max_epochs,
    #                          devices='auto',
    #                          strategy='auto',
    #                          logger=csvlogger1,
    #                          )

    #     trainer.fit(trainer_model, train_dataloader, ckpt_path=ckpt_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--loadckpt', type=str, default=None)
    parser.add_argument('--voxel_size', type=float)
    args = parser.parse_args()
    # if args.loadckpt is not None:
    #     train_resume(args.loadckpt)
    # else:
    train(args.voxel_size)
