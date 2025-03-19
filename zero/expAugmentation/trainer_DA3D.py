# framework package
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# zero package
from zero.expAugmentation.config.default import get_config, build_args
from zero.expAugmentation.dataset.dataset_DP_use_obsprocessor import Dataset_DP_PTV3
from zero.expAugmentation.models.DiffuserActor3D.Policy import Policy_Original
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

# ---------------------------------------------------------------
# region 0.Some tools


torch.set_float32_matmul_precision('medium')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 1. Trainer


class TrainerDP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.policy = Policy_Original(config['Policy'])

    def training_step(self, batch, batch_idx):
        loss = self.policy.forward(batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # 1. default optimizer
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.Train.lr)
        return optimizer


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 2. DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        train_data_path = os.path.join(self.config.B_Preprocess, 'train')
        val_data_path = os.path.join(self.config.B_Preprocess, 'val')
        train_dataset = Dataset_DP_PTV3(self.config, data_dir=train_data_path)
        val_dataset = Dataset_DP_PTV3(self.config, data_dir=val_data_path)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    def train_dataloader(self):
        batch_size = self.config.Train.batch_size
        sampler = DistributedSampler(self.train_dataset, shuffle=self.config.Train.shuffle,) if self.config.Train.num_gpus > 1 else None

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.config.Train.n_workers,
            pin_memory=self.config.Train.pin_mem,
            collate_fn=self.train_dataset.obs_processor.get_collect_function(),
            sampler=sampler,
            drop_last=False,
            prefetch_factor=2 if self.config.Train.n_workers > 0 else None,
            shuffle=self.config.Train.shuffle,
            persistent_workers=True
        )
        return loader

    def val_dataloader(self):
        batch_size = self.config.batch_size
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.config.num_gpus > 1 else None
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=self.config.Train.n_workers,
            pin_memory=self.config.Train.pin_mem,
            collate_fn=self.val_dataset.obs_processor.get_collect_function(),
            sampler=sampler,
            drop_last=False,
            prefetch_factor=2 if self.config.Train.n_workers > 0 else None,
            shuffle=self.config.Train.shuffle,
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


def train(config: yacs.config.CfgNode):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_name = current_time + '_DP'
    log_path = "/data/zero/2_Train/DP"
    log_name = ckpt_name
    # 1.trainer
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=200,
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
                         max_epochs=config.Train.epochs,
                         devices='auto',
                         strategy='auto',
                         logger=csvlogger1,
                         #  profiler=profiler，
                         #  profiler='simple',
                         use_distributed_sampler=False,
                         )
    config.freeze()
    trainer_model = TrainerDP(config)
    data_module = MyDataModule(config)
    trainer.fit(trainer_model, datamodule=data_module)
    # print(profiler.key_averages().table(max_len=200))


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 0.1 args & 0.2 config
    config_path = '/media/jian/ssd4t/zero/zero/expAugmentation/config/DA3D_Original.yaml'
    config = build_args(config_path)
    # 1. train
    train(config)
