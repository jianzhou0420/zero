# framework package
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# zero package
from zero.expForwardKinematics.config.default import get_config, build_args
from zero.expForwardKinematics.dataset.dataset_FK import DatasetFK as Dataset
from zero.expForwardKinematics.dataset.dataset_FK import collect_fn
from zero.expForwardKinematics.models.FK.Policy import PolicyFK
from zero.z_utils import *

# helper package
import time
from datetime import datetime
import yacs.config
import os
import warnings
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# ---------------------------------------------------------------
# region 0.Some tools
warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")

torch.set_float32_matmul_precision('medium')


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 1. Trainer


class Trainer_DP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.policy = PolicyFK(config)

    def training_step(self, batch, batch_idx):
        loss = self.policy.forward(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.policy.forward(batch)
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
        data_dir = self.config['TrainDataset']['data_dir']
        train_data_path = os.path.join(data_dir, 'train')
        val_data_path = os.path.join(data_dir, 'val')

        train_dataset = Dataset(self.config, data_dir=train_data_path)
        val_dataset = Dataset(self.config, data_dir=val_data_path)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        batch_size = self.config['Trainer']['train']['batch_size']
        sampler = DistributedSampler(self.train_dataset, shuffle=self.config['Trainer']['train']['shuffle'],) if self.config['Trainer']['num_gpus'] > 1 else None

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.config['Trainer']['train']['n_workers'],
            pin_memory=self.config['Trainer']['train']['pin_mem'],
            collate_fn=collect_fn,
            sampler=sampler,
            drop_last=False,
            # prefetch_factor=2 if self.config['Trainer']['n_workers'] > 1 else 0,
            shuffle=self.config['Trainer']['train']['shuffle'],
            persistent_workers=True
        )
        return loader

    def val_dataloader(self):
        batch_size = self.config['Trainer']['val']['batch_size']
        sampler = DistributedSampler(self.train_dataset, shuffle=self.config['Trainer']['val']['shuffle'],) if self.config['Trainer']['num_gpus'] > 1 else None

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=self.config['Trainer']['val']['n_workers'],
            pin_memory=self.config['Trainer']['val']['pin_mem'],
            collate_fn=collect_fn,
            sampler=sampler,
            drop_last=False,
            # prefetch_factor=2 if self.config['Trainer']['n_workers'] > 1 else 0,
            shuffle=self.config['Trainer']['val']['shuffle'],
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
    model_name = config['Trainer']['model_name']
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_name = current_time + model_name
    log_path = f"/data/zero/2_Train/{model_name}"
    log_name = ckpt_name
    # 1.trainer
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config['Trainer']['save_every_n_epochs'],
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
                         max_epochs=config['Trainer']['epoches'],
                         devices='auto',
                         strategy='auto',
                         logger=csvlogger1,
                         #  profiler=profiler，
                         #  profiler='simple',
                         use_distributed_sampler=False,
                         )
    config.freeze()
    trainer_model = Trainer_DP(config)
    data_module = MyDataModule(config)
    trainer.fit(trainer_model, datamodule=data_module)
    # print(profiler.key_averages().table(max_len=200))


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    # 0.1 args & 0.2 config
    pl.seed_everything(42)
    config_path = '/data/zero/zero/expForwardKinematics/config/FK.yaml'
    config = build_args(config_path)
    # 1. train
    train(config)
