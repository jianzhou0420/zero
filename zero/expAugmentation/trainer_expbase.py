# framework package

# own package
from pytorch_lightning.profilers import PyTorchProfiler
from zero.expAugmentation.models.lotus.optim.misc import build_optimizer
from zero.expAugmentation.dataset.dataset_expbase_voxel_augment import SimplePolicyDataset, ptv3_collate_fn
from zero.expAugmentation.models.lotus.model_expbase import SimplePolicyPTV3CA
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
    tasks_to_use: list = None
    num_gpu: int


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
        print('each_batch_idx:', batch_idx)
        print(f"each_step_allocated_cache: {torch.cuda.memory_allocated()/1024/1024/1024} GB")
        print(f"each_step_reserved_cache: {torch.cuda.memory_reserved()/1024/1024/1024} GB")
        print('dataids', batch['data_ids'])
        # del batch['pc_centroids'], batch['pc_radius']
        if batch_idx % (int(100 / (self.config.TRAIN.train_batch_size * self.config.num_gpu))) == 0:

            # print('batch_idx:', batch_idx)
            print('cache_empty')
            torch.cuda.empty_cache()
            # print(f"After empty cache: {torch.cuda.memory_allocated()} bytes")
            # print(f"After empty cache: {torch.cuda.memory_reserved()} bytes")
        losses = self.model(batch, is_train=True)
        self.log('train_loss', losses['total'], batch_size=len(batch['data_ids']), on_step=True, on_epoch=True, prog_bar=True, logger=True)        # if self.global_step % 10 == 0:
        # if self.global_step % 10 == 0:
        #     print(f"train_loss: {losses['total']}")
        #     print(f"train_loss: {losses['total']}")
        # print(f"After loading: {torch.cuda.memory_allocated()} bytes")
        return losses['total']

    def configure_optimizers(self):
        # optimizer, init_lrs = build_optimizer(self.model, self.config.TRAIN)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        # scheduler = WarmupCosineScheduler(
        #     optimizer,
        #     warmup_steps=self.config.TRAIN.warmup_steps,
        #     total_steps=self.config.TRAIN.num_train_steps,
        # )

        # scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",  # Adjust learning rate every step
        # }

        return [optimizer], [scheduler]

    def get_dataset(self, config):
        dataset = SimplePolicyDataset(config=config, is_single_frame=False, tasks_to_use=config.tasks_to_use, **config.TRAIN_DATASET)
        return dataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # Download or preprocess data if necessary
        pass

    def setup(self, stage=None):
        dataset = SimplePolicyDataset(config=self.config, is_single_frame=False, tasks_to_use=self.config.tasks_to_use, **self.config.TRAIN_DATASET)
        self.train_dataset = dataset

    def train_dataloader(self):

        batch_size = self.config.TRAIN.train_batch_size
        sampler = DistributedSampler(self.train_dataset, shuffle=False)

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


if __name__ == '__main__':
    def train(exp_name, config, current_time):
        ckpt_name = f'{current_time}_{exp_name}'

        log_path = config.log_dir
        log_name = f'{current_time}_{exp_name}'

        # 0. prepare config
        # check batch size and number of gpus
        bs = config.TRAIN.train_batch_size
        gpu = config.num_gpu
        assert 100 % (bs * gpu) == 0, "Batch size should be divisible by 100, please change the batch size or number of gpus"

        # num_train_steps
        epoches = config.TRAIN.epoches

        if config.tasks_to_use is not None:
            num_tasks = len(config.tasks_to_use)
            num_episodes = num_tasks * 100
            total_episodes = num_episodes * epoches
            total_steps = total_episodes // (bs * gpu)
        else:
            total_steps = 18 * epoches * 100 // (bs * gpu)

        config.TRAIN.num_train_steps = total_steps
        config.TRAIN.warmup_steps = total_steps // 15

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

        print(f"max_epochs: {config.TRAIN.epoches}")
        profiler = PyTorchProfiler(
            output_filename="profiler_output.json",  # This will output a JSON trace.
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # Optionally set up a schedule to control when profiling is active:
            schedule=torch.profiler.schedule(
                wait=10,    # number of steps to skip before warming up
                warmup=20,  # number of warmup steps
                active=2,  # number of steps to profile
                repeat=1   # number of times to repeat the cycle
            ),

            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,   # Optionally record tensor shapes for more insight.
        )

        trainer = pl.Trainer(callbacks=[checkpoint_callback],
                             max_epochs=config.TRAIN.epoches,
                             devices='auto',
                             strategy='ddp',
                             logger=csvlogger1,
                             #  profiler=profiler，
                             profiler='simple',
                             use_distributed_sampler=False
                             )
        trainer_model = TrainerLotus(config)
        data_module = MyDataModule(config)
        trainer.fit(trainer_model, datamodule=data_module)
        # print(profiler.key_averages().table(max_len=200))

    # 0.1 args
    args = TrainerArgs().parse_args(known_only=True)
    # 0.2 config
    config = yacs.config.CfgNode(new_allowed=True)

    args_dict = {}
    for arg in args.class_variables.keys():
        args_dict[arg] = getattr(args, arg)

    for key, value in args_dict.items():
        config[key] = value

    config.merge_from_file(args.config)

    # 0.3 path & name stuff
    current_time = datetime.now().strftime("%Y_%m_%d__%H-%M")

    exp_name = args.name

    train(exp_name, config, current_time)
