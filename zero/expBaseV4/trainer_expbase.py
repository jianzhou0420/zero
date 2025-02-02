# framework package

# own package
from zero.expBaseV4.models.lotus.optim.misc import build_optimizer
from zero.expBaseV4.dataset.dataset_expbase_voxel import SimplePolicyDataset, ptv3_collate_fn
from zero.expBaseV4.models.lotus.model_expbase import SimplePolicyPTV3CA
from zero.z_utils import *

# python package
import re
import tap
from datetime import datetime
import yacs.config
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
import os
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger

import warnings

warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")

torch.set_float32_matmul_precision('medium')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class TrainerArgs(tap.Tap):
    config: str = None
    name: str = 'default'
    resume_version_dir: str = None


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
        self.save_hyperparameters()
        self.config = config
        self.model = SimplePolicyPTV3CA(config.MODEL)

    def training_step(self, batch, batch_idx):

        # del batch['pc_centroids'], batch['pc_radius']

        losses = self.model(batch, is_train=True)
        self.log('train_loss', losses['total'], batch_size=len(batch['data_ids']), on_step=True, on_epoch=True, prog_bar=True, logger=True)        # if self.global_step % 10 == 0:
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
        dataset = SimplePolicyDataset(config=config, is_single_frame=False, **config.TRAIN_DATASET)
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
                shuffle=False,
                persistent_workers=True
            )
            return loader
        # function
        dataset = self.get_dataset(config)
        train_loader = build_dataloader(dataset, ptv3_collate_fn, is_train=True, config=config)
        return train_loader


if __name__ == '__main__':
    def train(exp_name, config, current_time):
        ckpt_name = f'{current_time}_{exp_name}'

        log_path = config.log_dir
        log_name = f'{current_time}_{exp_name}'

        # 1.trainer
        trainer_model = TrainerLotus(config)

        train_dataloader = trainer_model.get_dataloader(config)

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

        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"max_epochs: {config.TRAIN.epoches}")

        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=config.TRAIN.epoches,
                             devices='auto',
                             strategy='auto',
                             logger=csvlogger1,
                             )

        trainer.fit(trainer_model, train_dataloader)

    def resume(exp_name, config, logs_dir):
        # logs_dir should navigate to the version folder
        version = logs_dir.split('/')[-1]
        log_name = logs_dir.split('/')[-2]
        log_path = "/".join(logs_dir.split('/')[0:-2])

        ckpt_folder = os.path.join(logs_dir, 'checkpoints')
        ckpts = sorted(os.listdir(ckpt_folder), key=natural_sort_key)

        ckpt_path = os.path.join(ckpt_folder, ckpts[-1])
        ckpt_name = ckpts[-1].split('epoch=')[0]

        # print(f"ckpt_path: {ckpt_path}")
        # print(f"ckpt_name: {ckpt_name}")
        # print(f"log_path: {log_path}")
        # print(f"log_name: {log_name}")

        csvlogger1 = CSVLogger(
            save_dir=log_path,
            name=log_name,
            version=version
        )

        trainer_model = TrainerLotus(config)
        train_dataloader = trainer_model.get_dataloader(config)
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=100,
            save_top_k=-1,
            save_last=False,
            filename=f'{ckpt_name}' + '{epoch:03d}'  # Checkpoint filename
        )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=config.TRAIN.epoches,
            devices='auto',
            strategy='auto',
            logger=csvlogger1,
            # resume_from_checkpoint=ckpt_path  # Resume from this checkpoint if provided
        )

        trainer.fit(trainer_model, train_dataloader, ckpt_path=ckpt_path)

        # 0.1 args
    args = TrainerArgs().parse_args(known_only=True)
    # 0.2 config
    config = yacs.config.CfgNode(new_allowed=True)
    config.merge_from_file(args.config)

    # 0.3 path & name stuff
    current_time = datetime.now().strftime("%Y_%m_%d__%H-%M")

    exp_name = args.name
    if args.resume_version_dir is None:
        train(exp_name, config, current_time)
    else:
        resume(exp_name, config, args.resume_version_dir)
