# framework package
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import os
import math
from argparse import Namespace
from typing import List, Dict, Tuple, Union, Iterator
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
import yacs.config
# utils package
import yaml
from datetime import datetime
import argparse
from zero.v1.models.lotus.simple_policy_ptv3 import SimplePolicyPTV3CA
from zero.v1.dataset.dataset_lotus_modified import SimplePolicyDataset, ptv3_collate_fn
from zero.v1.models.lotus.optim.misc import build_optimizer
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

        _, losses = self.model(batch, compute_loss=True, compute_final_action=False)
        self.log('train_loss', losses['total'])
        if self.global_step % 100 == 0:
            print(f"train_loss: {losses['total']}")
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
    def train():
        config = yacs.config.CfgNode(new_allowed=True)
        config.merge_from_file('/workspace/zero/zero/v1/config/lotus.yaml')

        trainer_model = TrainerLotus(config)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_dataloader = trainer_model.get_dataloader(config)

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=10,
            save_last=True,
            filename=f'{current_time}' + '{epoch:03d}'  # Checkpoint filename
        )
        csvlogger1 = CSVLogger('/data/ckpt/logs', name='csvlogger2')
        # scale_factor = 8 / config.TRAIN.train_batch_size
        # max_epochs = int(config.TRAIN.num_train_steps // len(train_dataloader) * scale_factor)
        max_epochs = int(1368)
        print(f"config.TRAIN.num_train_steps: {config.TRAIN.num_train_steps}")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"max_epochs: {max_epochs}")
        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=max_epochs,
                             devices='auto',
                             strategy='auto',
                             logger=csvlogger1,)

        trainer.fit(trainer_model, train_dataloader)

    train()
