# framework package
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F

# My core package
from zero.v1.dataset.datasets_18_tasks import JianRLBenchDataset
from zero.v1.models.zero_test import ZeroModel

# utils package
import yaml
from datetime import datetime
import argparse

'''
Jian: To make my code clean, this file only contain the code of trainning and evaluation.
      Details of the model can be found in CVAE.py
'''

torch.set_float32_matmul_precision('medium')


class TrainerTesterJazz(pl.LightningModule):
    """
    This is a template for a my implementation of PyTorch Lightning module.
    """

    def __init__(self, config):
        super().__init__()
        self.path = dict()

        self.path['dataset_train_path'] = config['trainer']['dataset_train_path']
        self.path['dataset_val_path'] = config['trainer']['dataset_val_path']
        self.path['ckpt_path'] = config['trainer']['ckpt_path']

        self.model = ZeroModel(config)
        self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00001,
                                           weight_decay=0.0001)

    ############################
    # Training_Helpers
    ############################

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, data_dict, idx):

        self.model.train()
        model_output = self._forward_pass(data_dict)
        data_dict['future_position'] = data_dict['future_position'].squeeze().cuda()
        loss = F.l1_loss(model_output, data_dict['future_position'])

        # log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=128)
        if idx % 5 == 0:
            print(f"train_loss: {loss},idx: {idx}")
        # /log
        return loss

    # def validation_step(self, data_dict, idx):
    #     return None

    def _forward_pass(self, data_dict):
        '''
        receive data and return the loss
        '''
        if data_dict['images'].shape[-1] == 3:
            data_dict['images'] = data_dict['images'].cuda().permute(0, 1, 4, 2, 3)

        model_output = self.model(data_dict['text'], data_dict['images'])

        return model_output

    def _get_dataloaders(self):
        train_dataset = JianRLBenchDataset(self.path['dataset_train_path'])
        # val_dataset = JianRLBenchDataset(self.path['dataset_val_path'])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
        # test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)
        return train_loader


if __name__ == '__main__':

    with open('/workspace/zero/zero/v1/config/zero_test.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    trainer_pl = TrainerTesterJazz(config)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dataloader = trainer_pl._get_dataloaders()

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_last=True,
        dirpath=trainer_pl.path['ckpt_path'],  # Directory to save the checkpoints
        filename=f'{current_time}' + '{epoch:03d}'  # Checkpoint filename
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         max_epochs=51,
                         devices=4,
                         strategy='ddp',
                         default_root_dir='/data/ckpt')
    trainer.fit(trainer_pl, train_dataloader)
