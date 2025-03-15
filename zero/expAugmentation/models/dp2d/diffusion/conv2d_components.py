import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange
import math


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        # debug
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        # /debut

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)


class Downsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv2dBlock(nn.Module):
    '''
        Conv2d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test2d():
    # Test Conv2dBlock
    cb = Conv2dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16, 16))
    o = cb(x)
    print(o.shape)

    # Test Downsample2d
    ds = Downsample2d(256)
    x = torch.zeros((1, 256, 16, 16))
    o = ds(x)
    print(o.shape)

    # Test Upsample2d
    us = Upsample2d(256)
    x = torch.zeros((1, 256, 16, 16))
    o = us(x)
    print(o.shape)


# test2d()
