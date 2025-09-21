# helper package
try:
    import warnings
    warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_bwd.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
except:
    pass

import time
# from datetime import datetime # No longer strictly needed if Hydra handles time-based naming
import os


# framework package
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger # Instantiated via Hydra
import pytorch_lightning as pl
from torch.utils.data import Dataset

# zero package
# from zero.expForwardKinematics.config.default import get_config, build_args # Replaced by Hydra
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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Keep if necessary for your environment
os.environ['TORCH_USE_CUDA_DSA'] = "1"  # Keep if necessary for your environment
torch.set_float32_matmul_precision('medium')

# POLICY_FACTORY, DATASET_FACTORY, CONFIG_FACTORY are replaced by Hydra's instantiation mechanism
# and config composition. Target classes should be specified in Hydra config YAMLs.

# ---------------------------------------------------------------
# region 1. Trainer


class Trainer_all(pl.LightningModule):
    def __init__(self, cfg: DictConfig):  # Full Hydra config
        super().__init__()
        # self.save_hyperparameters(cfg) # Saves the entire Hydra config, can be verbose
        # It's often better to save a container version or specific parts:
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        self.cfg = cfg
        self.policy = instantiate(cfg.policy)

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)
        for k, v in loss.items():
            self.log(f'train/{k}', v, prog_bar=True)
        return loss['total_loss']

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        loss = self.policy(batch)  # Assuming policy returns a dict also for validation
        # The original code logged `loss` directly, which might be a dict.
        # If `loss` is a dict from policy, and we need to log a specific scalar value for 'val_loss':
        if isinstance(loss, dict):
            # Assuming 'total_loss' or a similar key should be logged for validation
            # This part might need adjustment based on what self.policy(batch) returns for validation
            scalar_val_loss = loss.get('total_loss', None)  # Or another relevant key
            if scalar_val_loss is not None:
                self.log('val_loss', scalar_val_loss, prog_bar=True)
            else:  # Log all items in loss dict if val_loss specific key is not found
                for k, v in loss.items():
                    self.log(f'val/{k}', v, prog_bar=True)

        else:  # if loss is already a scalar
            self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer is instantiated using cfg.optimizer.
        # cfg.optimizer should have _target_ and hyperparameters like lr, weight_decay.
        # These hyperparams can refer to other parts of the config (e.g., lr: ${Trainer.lr})
        optimizer = instantiate(self.cfg.optimizer, params=self.policy.parameters())
        return optimizer


# endregion
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# region 2. DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):  # Full Hydra config
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        # Dataset is instantiated using cfg.dataset.
        # Assuming original DatasetClass(config) means DatasetClass.__init__(self, config) where 'config' is the full config object:
        self.train_dataset = instantiate(self.cfg.dataset, config=self.cfg)

    def train_dataloader(self):
        batch_size = self.cfg.Trainer.train.batch_size  # Accessing nested config

        # Sampler logic: PL's DDP strategy usually handles this.
        # If using DDP (num_gpus > 1), PL's DDP strategy with replace_sampler_ddp=True (default)
        # will replace the sampler. Explicitly creating DistributedSampler might be redundant
        # or conflict unless PL's default DDP sampler replacement is disabled.
        # The original code had use_distributed_sampler=False in pl.Trainer.
        sampler = None
        if self.cfg.Trainer.num_gpus > 1 and not self.trainer. przypadki_użycia_distributed_sampler:  # Check if PL is using its own sampler
            sampler = DistributedSampler(self.train_dataset, shuffle=self.cfg.Trainer.train.shuffle)

        print(f"batch_size: {batch_size}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.Trainer.train.n_workers,
            pin_memory=self.cfg.Trainer.train.pin_mem,
            sampler=sampler,
            drop_last=False,  # Original: drop_last=False
            shuffle=self.cfg.Trainer.train.shuffle if sampler is None else False,  # Shuffle is mutually exclusive with sampler
            persistent_workers=True if self.cfg.Trainer.train.n_workers > 0 else False,  # persistent_workers require num_workers > 0
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

# The 'train' function now takes Hydra's CfgNode (DictConfig)
def train(cfg: DictConfig):
    # model_name = cfg.Trainer.model_name # Or from a top-level cfg.model_name
    # current_time = datetime.now().strftime('%b%d_%H-%M-%S') # Hydra handles output dirs and can provide time via ${now:}

    # Loggers and Callbacks are instantiated from the config.
    # Their parameters (names, paths) should be defined in the YAML config,
    # ideally using Hydra's interpolation (e.g., ${hydra:runtime.output_dir}, ${now:%Y-%m-%d}).

    loggers_list = []
    if cfg.loggers:
        for logger_name, logger_conf in cfg.loggers.items():
            if logger_conf and logger_conf._target_:  # Check if logger is configured
                # Update save_dir for CSVLogger to be relative to Hydra's output if not absolute
                if 'CSVLogger' in logger_conf._target_ and 'save_dir' in logger_conf and not os.path.isabs(logger_conf.save_dir):
                    logger_conf.save_dir = f"{HydraConfig.get().runtime.output_dir}/{logger_conf.save_dir}"
                loggers_list.append(instantiate(logger_conf))

    callbacks_list = []
    if cfg.callbacks:
        for cb_name, cb_conf in cfg.callbacks.items():
            if cb_conf and cb_conf._target_:  # Check if callback is configured
                # Update dirpath for ModelCheckpoint to be relative to Hydra's output if not absolute
                if 'ModelCheckpoint' in cb_conf._target_ and 'dirpath' in cb_conf and not os.path.isabs(cb_conf.dirpath):
                    cb_conf.dirpath = f"{HydraConfig.get().runtime.output_dir}/{cb_conf.dirpath}"
                callbacks_list.append(instantiate(cb_conf))

    trainer_pl = instantiate(cfg.pytorch_lightning_trainer,
                             callbacks=callbacks_list,
                             logger=loggers_list,
                             )
    # config.freeze() # OmegaConf passed to @hydra.main is already frozen by default.

    # Instantiate the main training module (Trainer_all)
    # It receives the full config 'cfg' which it can use to instantiate sub-modules like the policy.
    trainer_model = instantiate(cfg.trainer_module, cfg=cfg)

    # Instantiate the data module
    data_module = instantiate(cfg.datamodule, cfg=cfg)

    trainer_pl.fit(trainer_model, datamodule=data_module)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Optional: Print the resolved config
    # print(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, 'seed') and cfg.seed is not None:
        pl.seed_everything(cfg.seed)
    else:
        # Fallback seed if not in config, or remove this if seed must be in config
        pl.seed_everything(42)

    train(cfg)


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
    main()
