# helper package
try:
    import warnings
    warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_bwd.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
except:
    pass

import math
from datetime import datetime
import yacs.config
import os
import os
from typing import Type, Dict
# framework package
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.loggers import WandbLogger
# zero package
from zero.expForwardKinematics.config.default import get_config, build_args
from zero.expForwardKinematics.models.FK.Policy import PolicyFK
from zero.z_utils import *
from zero.expForwardKinematics.models.DP.DP import DPWrapper
from zero.expForwardKinematics.models.DP3.DP3Wrapper import DP3Wrapper
from zero.expForwardKinematics.ObsProcessor import *
from zero.expForwardKinematics.dataset.dataset_general import DatasetGeneral
from zero.expForwardKinematics.dataset.dataset_DA3DWrapper import DA3DDatasetWrapper
from zero.expForwardKinematics.models.DA3DWrapper.DA3DWrapper import DA3DWrapper
from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP_traj import ObsProcessorDP_traj
from zero.expForwardKinematics.dataset.dataset_general_traj import DatasetGeneral_traj
from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP_traj_zarr import ObsProcessorDP_traj_zarr
from zero.expForwardKinematics.dataset.dataset_zarr import DatasetTmp
from zero.expForwardKinematics.models.DP.DP_newloss import DPWithLossWrapper
from zero.expForwardKinematics.ObsProcessor.ObsProcessorDP_traj_zarr_x0loss import ObsProcessorDP_traj_zarr_x0loss
from zero.expForwardKinematics.models.Base.BaseAll import BasePolicy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
torch.set_float32_matmul_precision('medium')

POLICY_FACTORY: Dict[str, Type[BasePolicy]] = {
    'FK': PolicyFK,
    'DP': DPWrapper,
    'DA3D': DA3DWrapper,
    'DP3': DP3Wrapper,
    'DP_traj': DPWrapper,
    'DP_traj_zarr': DPWrapper,
    'DP_traj_x0loss': DPWithLossWrapper
}


OBS_FACTORY: Dict[str, Type[ObsProcessorRLBenchBase]] = {
    'FK': ObsProcessorFK,
    'DP': ObsProcessorDP,
    'DP3': ObsProcessorDP3,
    'DA3D': ObsProcessorDA3DWrapper,
    'DP_traj': ObsProcessorDP_traj,
    'DP_traj_zarr': ObsProcessorDP_traj_zarr,
    'DP_traj_x0loss': ObsProcessorDP_traj_zarr_x0loss,
}

DATASET_FACTORY: Dict[str, Type[Dataset]] = {
    'FK': DatasetGeneral,
    'DP': DatasetGeneral,
    'DP3': DatasetGeneral,
    'DA3D': DA3DDatasetWrapper,
    'DP_traj': DatasetGeneral_traj,
    'DP_traj_zarr': DatasetTmp,
    'DP_traj_x0loss': DatasetTmp,
}


CONFIG_FACTORY = {
    'FK': './zero/expForwardKinematics/config/FK.yaml',
    'DP': './zero/expForwardKinematics/config/DP.yaml',
    'DP3': './zero/expForwardKinematics/config/DP3.yaml',
    'DA3D': './zero/expForwardKinematics/config/DA3DWrapper.yaml',
    'DP_traj': './zero/expForwardKinematics/config/DP_traj.yaml',
    'DP_traj_zarr': './zero/expForwardKinematics/config/DP_traj_zarr.yaml',
    'DP_traj_x0loss': './zero/expForwardKinematics/config/DP_traj_x0loss.yaml',
}


# ---------------------------------------------------------------
# region 1. Trainer
class Trainer_all(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        Policy = POLICY_FACTORY[config['Trainer']['model_name']]
        print(f"Policy: {Policy}")
        self.policy = Policy(config)

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        print(f"train_loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        loss = self.policy(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.policy.parameters(), weight_decay=self.config['Trainer']['weight_decay'], lr=self.config['Trainer']['lr'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
            'interval': 'step',  # or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]


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

        if os.path.exists(val_data_path) is False:
            self.use_val = False
        else:
            self.use_val = True

        obs_processor = OBS_FACTORY[self.config['Trainer']['model_name']]
        DatasetClass = DATASET_FACTORY[self.config['Trainer']['model_name']]

        collect_fn = obs_processor.collate_fn
        train_dataset = DatasetClass(self.config, data_dir=train_data_path, ObsProcessor=obs_processor)

        self.train_dataset = train_dataset
        self.collect_fn = collect_fn

        if self.use_val:
            val_dataset = DatasetClass(self.config, data_dir=val_data_path, ObsProcessor=obs_processor)
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
            collate_fn=self.collect_fn,
            sampler=sampler,
            drop_last=False,
            # prefetch_factor=2 if self.config['Trainer']['n_workers'] > 1 else 0,
            shuffle=self.config['Trainer']['train']['shuffle'] if sampler is None else None,
            persistent_workers=True
        )
        return loader

    def val_dataloader(self):
        if self.use_val is False:
            return []
        batch_size = self.config['Trainer']['val']['batch_size']
        sampler = DistributedSampler(self.train_dataset, shuffle=self.config['Trainer']['val']['shuffle'],) if self.config['Trainer']['num_gpus'] > 1 else None

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=self.config['Trainer']['val']['n_workers'],
            pin_memory=self.config['Trainer']['val']['pin_mem'],
            collate_fn=self.collect_fn,
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


def train(config: yacs.config.CfgNode):
    model_name = config['Trainer']['model_name']
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_name = current_time + model_name
    log_path = f"./2_Train/{model_name}"
    log_name = ckpt_name

    # 1.trainer
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=config['Trainer']['save_every_n_epochs'],
        save_top_k=-1,
        save_last=False,
        filename=f'{ckpt_name}_' + '{epoch:03d}',  # Checkpoint filename
        save_on_train_epoch_end=True,  # must have it if you config check_val_every_n_epoch on trainer
    )

    tflogger = TensorBoardLogger(
        save_dir=log_path,
        name=log_name,
        version=None
    )

    csvlogger = CSVLogger(
        save_dir=log_path,
        name=log_name,
        version=None
    )

    # wandblogger = WandbLogger(
    #     project='DecoupleActionHead',
    #     name=log_name,
    #     save_dir=log_path,
    #     log_model=True,
    # )
    num_gpus = torch.cuda.device_count()
    # num_gpus = 2
    config['Trainer']['num_gpus'] = num_gpus

    print(f"num_gpus: {num_gpus}")
    strategy = DDPStrategy(find_unused_parameters=True) if num_gpus > 1 else 'auto'

    trainer = pl.Trainer(callbacks=[checkpoint_callback, EpochCallback()],
                         #  max_steps=config['Trainer']['max_steps'],
                         max_epochs=config['Trainer']['epoches'],
                         devices='auto',
                         strategy=strategy,
                         logger=[csvlogger, tflogger],
                         #  profiler=profiler，
                         #  profiler='simple',
                         use_distributed_sampler=True if num_gpus > 1 else False,
                         check_val_every_n_epoch=config['Trainer']['check_val_every_n_epoch'],
                         )
    config.freeze()
    trainer_model = Trainer_all(config)
    data_module = MyDataModule(config)
    trainer.fit(trainer_model, datamodule=data_module)


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train FK')
    argparser.add_argument('--model', type=str, required=True, help='model name')
    argparser.add_argument('--config', type=str, default=None, help='config file path')
    args = argparser.parse_args()
    pl.seed_everything(42)

    if args.config is not None:
        config_path = args.config
    else:
        config_path = CONFIG_FACTORY[args.model]

    config = get_config(config_path)
    # 1. train
    train(config)
