# src/train_hydra.py

from zero.expJPeePose.model.mlp import MLP
from zero.expJPeePose.model.VAE import VAE
from zero.expJPeePose.dataset.dataset_math import DatasetGeneral
from zero.expForwardKinematics.ObsProcessor import *
from zero.z_utils import *
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Type, Dict
import os
import warnings
warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
warnings.filterwarnings("ignore", category=FutureWarning)


# Framework packages

# Hydra and OmegaConf

# Your project's packages

# Setup environment and torch settings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
torch.set_float32_matmul_precision('medium')


# --- FACTORIES (remains the same) ---
POLICY_FACTORY = {'VAE': VAE, 'MLP': MLP}
DATASET_FACTORY: Dict[str, Type[Dataset]] = {'VAE': DatasetGeneral, 'MLP': DatasetGeneral}


# ---------------------------------------------------------------
# region 1. LightningModule (Modified for DictConfig)
class Trainer_all(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Using save_hyperparameters with the config makes it available in checkpoints
        # and automatically logs to W&B
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        Policy = POLICY_FACTORY[self.cfg.model.model_name]
        print(f"Using Policy: {Policy.__name__}")
        self.policy = Policy(cfg)  # Pass the whole config to the policy

    def training_step(self, batch, batch_idx):
        loss = self.policy(batch)
        for k, v in loss.items():
            self.log(f'train/{k}', v, prog_bar=(k == 'total_loss'))
        return loss['total_loss']

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        # The original code logged a dictionary, which is not standard.
        # Assuming the policy returns a dictionary of losses similar to training_step.
        loss_dict = self.policy(batch)
        val_loss = loss_dict.get('total_loss') or loss_dict.get('loss')
        if val_loss is not None:
            self.log('val_loss', val_loss, prog_bar=True)
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.cfg.Trainer.lr,
            weight_decay=self.cfg.Trainer.weight_decay
        )
        return optimizer

# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 2. DataModule (Modified for DictConfig)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None

    def setup(self, stage=None):
        DatasetClass = DATASET_FACTORY[self.cfg.model.model_name]
        self.train_dataset = DatasetClass(self.cfg)

    def train_dataloader(self):
        # Note: DistributedSampler is handled automatically by PyTorch Lightning
        # when using strategy='ddp' or similar. No need for manual sampler.
        print(f"Batch size: {self.cfg.Trainer.train.batch_size}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.Trainer.train.batch_size,
            num_workers=self.cfg.Trainer.train.n_workers,
            pin_memory=self.cfg.Trainer.train.pin_mem,
            shuffle=self.cfg.Trainer.train.shuffle,
            persistent_workers=(self.cfg.Trainer.train.n_workers > 0)
        )
        return loader

# endregion
# ---------------------------------------------------------------

# An empty callback if you need it for future use


class EpochCallback(Callback):
    pass

# ---------------------------------------------------------------
# region Main Training Function (Rewritten for Hydra and W&B)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    # Print the configuration - useful for debugging
    print(OmegaConf.to_yaml(cfg))

    # 1. Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # 2. Setup W&B Logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.model.model_name}-{os.path.basename(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)}",
        log_model=cfg.wandb.log_model,
        # The config is automatically logged when passed to save_hyperparameters
    )

    # 3. Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoints"),  # Save checkpoints in the hydra output dir
        filename=f"{cfg.model.model_name}-{{epoch:03d}}",
        every_n_epochs=cfg.Trainer.save_every_n_epochs,
        save_top_k=-1,  # Save all checkpoints
        save_on_train_epoch_end=True,
    )

    # 4. Instantiate Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.Trainer.epoches,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, EpochCallback()],
        devices='auto',
        strategy='auto',
        check_val_every_n_epoch=cfg.Trainer.check_val_every_n_epoch
    )

    # 5. Instantiate Model and DataModule
    model = Trainer_all(cfg)
    datamodule = MyDataModule(cfg)

    # 6. Start Training
    trainer.fit(model, datamodule=datamodule)

    # 7. Finish W&B Run
    # This is important to ensure all data is synced.
    wandb_logger.experiment.finish()


# endregion
# ---------------------------------------------------------------

if __name__ == '__main__':
    train()
