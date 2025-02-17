# framework package
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# zero package
from .config.default import get_config
from .models.lotus.optim.misc import build_optimizer
from .dataset.dataset_expbase_voxel_augment import LotusDatasetAugmentation, ptv3_collate_fn
from .dataset.dataset_expbase_voxel import LotusDataset, ptv3_collate_fn
from .models.lotus.model_expbase import SimplePolicyPTV3CA
from zero.z_utils import *

# helper package
import argparse
import time
import re
from datetime import datetime
import yacs.config
import math
import os
import warnings
warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
torch.set_float32_matmul_precision('medium')

# ---------------------------------------------------------------
# region 0.Some tools


torch.set_float32_matmul_precision('medium')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


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
# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 1. Trainer


class TrainerLotus(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = SimplePolicyPTV3CA(config.MODEL)

    def training_step(self, batch, batch_idx):

        if batch_idx % (int(100 / (self.config.batch_size * self.config.num_gpus))) == 0:
            # print('batch_idx:', batch_idx)
            print('At batch_idx:', batch_idx, 'each_step_allocated_cache:', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 'GB')
            print('At batch_idx:', batch_idx, 'each_step_reserved_cache:', torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 'GB')
            torch.cuda.empty_cache()
            print('At batch_idx:', batch_idx, 'each_step_allocated_cache:', torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 'GB')
            print('At batch_idx:', batch_idx, 'each_step_reserved_cache:', torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 'GB')

        losses = self.model(batch, is_train=True)
        self.log('train_loss', losses['total'], batch_size=len(batch['data_ids']), on_step=True, on_epoch=True, prog_bar=True, logger=True)        # if self.global_step % 10 == 0:
        if self.global_step % 10 == 0:
            print(f"train_loss: {losses['total']}")
        # print(f"train_loss: {losses['total']}")
        # print(f"After loading: {torch.cuda.memory_allocated()} bytes")
        return losses['total']

    def configure_optimizers(self):

        if self.config.optimizer == 'default':
            # 1. default optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
            return [optimizer], [scheduler]

        elif self.config.optimizer == 'lotus':

            # 2. lotus custom optimizer
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


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 2. DataModule


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # Download or preprocess data if necessary
        pass

    def setup(self, stage=None):
        if config.dataset == 'lotus':
            dataset = LotusDataset(config=self.config, is_single_frame=False, tasks_to_use=self.config.tasks_to_use, **self.config.TRAIN_DATASET)
        elif config.dataset == 'augment':
            dataset = LotusDatasetAugmentation(config=self.config, is_single_frame=False, tasks_to_use=self.config.tasks_to_use, **self.config.TRAIN_DATASET)
        self.train_dataset = dataset

    def train_dataloader(self):
        batch_size = self.config.batch_size
        sampler = DistributedSampler(self.train_dataset, shuffle=False) if self.config.num_gpus > 1 else None

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=config.TRAIN.n_workers,
            pin_memory=config.TRAIN.pin_mem,
            collate_fn=ptv3_collate_fn,
            sampler=sampler,
            drop_last=False,
            prefetch_factor=2 if config.TRAIN.n_workers > 0 else None,
            shuffle=False,
            persistent_workers=True
        )
        return loader
# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 3. Callback


class EpochCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {trainer.current_epoch} took {epoch_time:.2f} seconds")
        trainer.logger.log_metrics({'epoch_time': epoch_time}, step=trainer.global_step)
# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region Main


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (use , to separate values in a list)",
    )
    args = parser.parse_args()

    config = get_config(args.exp_config, args.opts)
    return config


def train(config: yacs.config.CfgNode):
    config.defrost()
    current_time = datetime.now().strftime("%Y_%m_%d__%H-%M")
    exp_name = config.name
    ckpt_name = f'{current_time}_{exp_name}'
    log_path = config.log_dir
    log_name = f'{current_time}_{exp_name}'

    # 0. prepare config
    # check batch size and number of gpus
    bs = config.batch_size
    gpu = config.num_gpus
    assert 100 % (bs * gpu) == 0, "Batch size should be divisible by 100, please change the batch size or number of gpus"

    # num_train_steps
    epoches = config.epoches

    if config.tasks_to_use is not None:
        num_tasks = len(config.tasks_to_use)
        num_episodes = num_tasks * 100
        total_episodes = num_episodes * epoches
        total_steps = total_episodes // (bs * gpu)
    else:
        total_steps = 18 * epoches * 100 // (bs * gpu)

    config.TRAIN.num_train_steps = total_steps
    config.TRAIN.warmup_steps = total_steps // 15
    print(config.tasks_to_use)
    print(f"Total steps: {total_steps}, Warmup steps: {config.TRAIN.warmup_steps}")
    # 1.trainer
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=100,
        save_top_k=-1,
        save_last=False,
        filename=f'{ckpt_name}_' + '{epoch:03d}'  # Checkpoint filename
    )
    csvlogger1 = CSVLogger(
        save_dir=log_path,
        name=log_name,
        version=None
    )
    epoch_callback = EpochCallback()
    trainer = pl.Trainer(callbacks=[checkpoint_callback, epoch_callback],
                         max_epochs=config.epoches,
                         devices='auto',
                         strategy='ddp' if config.num_gpus > 1 else 'auto',
                         logger=csvlogger1,
                         #  profiler=profiler，
                         #  profiler='simple',
                         use_distributed_sampler=False,
                         precision=16 if config.fp16 else None,
                         )
    config.freeze()
    trainer_model = TrainerLotus(config)
    data_module = MyDataModule(config)
    trainer.fit(trainer_model, datamodule=data_module)
    # print(profiler.key_averages().table(max_len=200))


def resume():
    pass


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 0.1 args & 0.2 config
    config = build_args()
    # 1. train
    train(config)
