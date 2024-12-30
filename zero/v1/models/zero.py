

import torch
import torch.nn as nn
import torch.nn.functional as F


# 代码分成几个class，Trainer，ZeroModel，ActionHead,backbone.结构是Trainer.ZeroModel.[ActionHead,backbone]


class ActionHead(nn.Module):
    def __init__(self, config):
        super(ActionHead, self).__init__()
        self.action_head = nn.Linear(config['d_model'], config['n_actions'])

    def forward(self, x):
        return self.action_head(x)


class ZeroModel(nn.Module):
    def __init__(self, config):
        super(ZeroModel, self).__init__()
        self.transformer = DecoderOnlyTransformer(config)
        self.action_head = ActionHead(config)

    def forward(self, x):
        x = self.transformer(x)
        return self.action_head(x)
