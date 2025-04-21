from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange


from zero.expForwardKinematics.models.DP.diffusion.conv2d_components import Downsample2d, Upsample2d, Conv2dBlock
from zero.expForwardKinematics.models.DP.diffusion.pe import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class ConditionalResidualBlock2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv2dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv2dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2

        self.cond_predict_scale = cond_predict_scale

        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        self.timestep_encoder = nn.Sequential
        (
            nn.Mish(),
            nn.Linear(1, cond_channels),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, timestep, cond):
        '''
            x: (batch_size, in_channels, horizon, action)
            timestep: (batch_size,1)
            cond: (batch_size, cond_dim, horizon, action)
            out : [ batch_size , out_channels , horizon ,action]
        '''
        out = self.blocks[0](x)

        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet2D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False
                 ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim

        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock2D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock2D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv2dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv2d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                global_cond=None, **kwargs):
        """
        sample : (B, D, H, A ) bs, input_dim, horizon, action
        timestep: (B,)

        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """

        # 0. time
        timesteps = timestep
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


def test():
    policy = ConditionalUnet2D(input_dim=1,
                               global_cond_dim=256,
                               diffusion_step_embed_dim=256,
                               down_dims=[256, 512, 1024],
                               kernel_size=3,
                               n_groups=8,
                               cond_predict_scale=True)

    x = torch.zeros((1, 1, 8, 8))

    timestep = torch.zeros(1)

    global_cond = torch.zeros((1, 256, 8, 8))

    o = policy(x, timestep, global_cond)

    pass


test()
