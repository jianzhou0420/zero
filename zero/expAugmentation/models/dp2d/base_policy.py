import torch
import torch.nn as nn
'''
Code Structure Outlines
                      -- Clean but informative
'''

'''
ActionHead and Policy seems to have the same structure, but ActionHead should have more specific methods like q_sample, p_sample in DDPM.
Policy should have more general methods like train_one_step, inference_one_sample.
'''


class BaseActionHead(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def train_one_step(self,):
        loss = NotImplementedError, 'need to be implemented'
        return loss

    def inference_one_sample(self,):
        action = NotImplementedError, 'need to be implemented'
        return action


class BaseFeatureExtractor(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,):
        return NotImplementedError


class BasePolicy(nn.Module):
    def __init__(self,):
        super().__init__()
        self.ActionHead = BaseActionHead()
        self.FeatureExtractor = BaseFeatureExtractor()

    def train_one_step(self, batch):
        example_data = batch['data']
        features = self.FeatureExtractor(example_data)
        loss = self.ActionHead.train_one_step(features,)
        return loss

    def inference_one_sample(self, batch):
        example_data = batch['data']
        features = self.FeatureExtractor(example_data)
        action = self.ActionHead.inference_one_sample(features,)
        return action
