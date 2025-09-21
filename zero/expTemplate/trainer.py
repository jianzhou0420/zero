# helper package
try:
    import warnings
    warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_bwd.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
except:
    pass

import time
from datetime import datetime
import os
import os
from typing import Type, Dict
# framework package
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset
# zero package
from zero.expForwardKinematics.config.default import get_config, build_args
from zero.z_utils import *
from zero.expForwardKinematics.ObsProcessor import *
from zero.expTemplate.dataset.dataset_math import DatasetGeneral
from zero.expTemplate.model.VAE import VAE
from zero.expTemplate.model.mlp import MLP

# Hydra specific imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

torch.set_float32_matmul_precision('medium')


# ---------------------------------------------------------------
# region 1. Trainer
class Trainer_all(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        Policy = hydra.utils.instantiate(cfg.policy)

        self.policy = Policy

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)
        for k, v in loss.items():
            self.log(f'train/{k}', v, prog_bar=True)
        return loss['total_loss']

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        loss = self.policy(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.policy.parameters(), weight_decay=self.config['Trainer']['weight_decay'], lr=self.config['Trainer']['lr'])
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
        DatasetClass = DATASET_FACTORY[self.config['Trainer']['model_name']]
        train_dataset = DatasetClass(self.config)
        self.train_dataset = train_dataset

    def train_dataloader(self):
        batch_size = self.config['Trainer']['train']['batch_size']
        sampler = DistributedSampler(self.train_dataset, shuffle=self.config['Trainer']['train']['shuffle'],) if self.config['Trainer']['num_gpus'] > 1 else None

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.config['Trainer']['train']['n_workers'],
            pin_memory=self.config['Trainer']['train']['pin_mem'],
            sampler=sampler,
            drop_last=False,
            # prefetch_factor=2 if self.config['Trainer']['n_workers'] > 1 else 0,
            shuffle=self.config['Trainer']['train']['shuffle'],
            persistent_workers=True
        )
        return loader


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 3. Callback


class EpochCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # self.epoch_start_time = time.time()
        pass

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pass
        # epoch_time = time.time() - self.epoch_start_time
        # print(f"Epoch {trainer.current_epoch} took {epoch_time:.2f} seconds")
        # trainer.logger.log_metrics({'epoch_time': epoch_time}, step=trainer.global_step)
# endregion
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# region Main

@hydra.main(version_base=None, config_path="config", config_name="MLP")
def main(cfg: DictConfig) -> None:
    model_name = cfg['Trainer']['model_name']
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_name = current_time + model_name
    log_path = f"./2_Train/{model_name}"
    log_name = ckpt_name

    # 1.trainer
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=cfg['Trainer']['save_every_n_epochs'],
        save_top_k=-1,
        save_last=False,
        filename=f'{ckpt_name}_' + '{epoch:03d}',  # Checkpoint filename
        save_on_train_epoch_end=True,  # must have it if you cfg check_val_every_n_epoch on trainer
    )
    wandb_logger = WandbLogger(
        **cfg['Trainer']['wandb'],
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback, EpochCallback()],
                         #  max_steps=cfg['Trainer']['max_steps'],
                         max_epochs=cfg['Trainer']['epoches'],
                         devices='auto',
                         strategy='auto',
                         logger=[wandb_logger],
                         #  profiler=profiler，
                         #  profiler='simple',
                         use_distributed_sampler=False,
                         check_val_every_n_epoch=cfg['Trainer']['check_val_every_n_epoch'],
                         )
    trainer_model = Trainer_all(cfg)
    data_module = MyDataModule(cfg)
    trainer.fit(trainer_model, datamodule=data_module)
    # print(profiler.key_averages().table(max_len=200))


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    main()
