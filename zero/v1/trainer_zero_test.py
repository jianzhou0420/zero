# framework package
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F

# My core package
from zero.v1.dataset.datasets_18_tasks import JianRLBenchDataset
from zero.v1.models.zero_test import ZeroModel
from zero.v1.dataset.dataset_pusht import JianPushTDataset

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
    def __init__(self, config_first_layer):
        super().__init__()
        self.config = config_first_layer
        if config_first_layer['dataset_name'] == 'rlbench':
            config = config_first_layer['dataset_rlbench']
        elif config_first_layer['dataset_name'] == 'pusht':
            config = config_first_layer['dataset_pusht']

        self.path = dict()
        self.model = ZeroModel(config)
        self.model.cuda()

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00001,
                                           weight_decay=0.0001)
        return self.optimizer

    def training_step(self, data_dict, idx):
        if self.config['dataset_name'] == 'rlbench':
            self.model.train()
            model_output = self._forward_pass_rlbench(data_dict)
            data_dict['future_position'] = data_dict['future_position'].squeeze().cuda()
            loss = F.l1_loss(model_output, data_dict['future_position'])
            # log
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=config['dataset_rlbench']['batch_size'])
            # /log

        elif self.config['dataset_name'] == 'pusht':
            self.model.train()
            model_output = self._forward_pass_pusht(data_dict)
            loss = F.l1_loss(model_output, data_dict['action'].squeeze().cuda())
            # log
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=config['dataset_pusht']['batch_size'])
            # /log
        return loss

    #######################################
    # internal methods
    #######################################

    def _forward_pass_rlbench(self, data_dict):
        '''
        receive data and return the loss
        '''
        if data_dict['images'].shape[-1] == 3:
            data_dict['images'] = data_dict['images'].cuda().permute(0, 1, 4, 2, 3)
        model_output = self.model(data_dict['text'], data_dict['images'])
        return model_output

    def _forward_pass_pusht(self, data_dict):
        '''
        receive data and return the loss
        '''
        text = 'push T to the right place'
        text = [text for _ in range(data_dict['image'].shape[0])]
        images = data_dict['image'].cuda()
        images = images.unsqueeze(1)
        model_output = self.model(text, images)

        return model_output

    #######################################
    # interface methods
    #######################################

    def get_dataloaders(self):
        # get name
        dataset_name = self.config['dataset_name']

        # get config
        if dataset_name == 'rlbench':
            dataset_config = self.config['dataset_rlbench']
        elif dataset_name == 'pusht':
            dataset_config = self.config['dataset_pusht']

        dataset_train_path = dataset_config['dataset_train_path']
        bs = dataset_config['batch_size']
        shuffle = dataset_config['shuffle']
        num_workers = dataset_config['num_workers']

        # get dataset class and dataloader
        if dataset_name == 'rlbench':
            train_dataset = JianRLBenchDataset(dataset_train_path)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        elif dataset_name == 'pusht':
            train_dataset = JianPushTDataset(dataset_train_path)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)
        return train_loader


if __name__ == '__main__':
    with open('/workspace/zero/zero/v1/config/zero_test.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    trainer_pl = TrainerTesterJazz(config)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dataloader = trainer_pl.get_dataloaders()

    dataset_name = config['dataset_name']
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        save_last=True,
        filename=f'{current_time}' + f'{dataset_name}' + '{epoch:03d}'  # Checkpoint filename
    )

    csvlogger1 = CSVLogger('/data/ckpt/logs', name=f'{current_time}' + f'{dataset_name}')
    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         max_epochs=51,
                         devices=1,
                         strategy='auto',
                         default_root_dir='/data/ckpt',
                         logger=csvlogger1)

    trainer.fit(trainer_pl, train_dataloader)
