import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Type, List

ACTIVATION_FACTORY: Dict[str, Type[nn.Module]] = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'softmax': nn.Softmax,
}


class MLP(nn.Module):
    def __init__(self, config):
        """
        Args:
            layer_dims (list of int): e.g. [8, 16, 32, 64, 32].
            activation (nn.Module): activation class to use between layers.
            activate_last (bool): whether to apply activation after the final layer.
        """
        super().__init__()
        layers = []
        activation = ACTIVATION_FACTORY[config['Model']['activation']]
        layer_dims = config['Model']['middle_dims']  # type: List
        activate_last = config['Model']['activate_last']
        if config['Model']['FK'] is True:
            layer_dims.insert(0, 7)  # JP 7
            layer_dims.append(9)  # PosOrtho6D 3+6
        else:
            layer_dims.insert(0, 9)
            layer_dims.append(7)  # eePose 3+4

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            # apply activation after every layer except, by default, the last
            layers.append(activation(inplace=True))
        if not activate_last:
            # remove the final activation
            layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        input = batch['input']
        output = batch['output']
        x = self.net(input)
        loss = F.mse_loss(x, output)

        return {
            'total_loss': loss,
        }

    def inference_one_sample(self, batch, denormalize=False):
        input = batch['input']
        x = self.net(input)
        if denormalize:
            x = x * batch['output_std'] + batch['output_mean']
        return x


if __name__ == "__main__":
    # define your architecture
    dims = [8, 16, 32, 64, 32]
    model = MLP(dims)

    # test with dummy data
    x = torch.randn(5, dims[0])   # batch_size=5, input_dim=8
    y = model(x)
    print(y.shape)                # -> torch.Size([5, 32])
