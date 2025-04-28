from diffusion_policy_3d.model.common.normalizer import SingleFieldLinearNormalizer
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from termcolor import cprint


from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

from diffusion_policy_3d.policy.dp3 import DP3

from omegaconf import OmegaConf, DictConfig


class DP3Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        shape_meta = OmegaConf.create({
            "obs": {
                "point_cloud": {
                    "shape": [4096, 3],
                    "type": "rgb",
                },
                "agent_pos": {
                    "shape": [8],
                    "type": "low_dimx",
                },
                # "eePos": {
                #     "shape": [3],
                #     # type default: low_dim
                # },
                # "eeRot": {
                #     "shape": [4],
                # },
                # "eeOpen": {
                #     "shape": [1],
                # },
            },
            "action": {
                "shape": [8],
            },
        })
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample",
        )

        pointcloud_endoer_cfg = OmegaConf.create({
            'in_channels': 3,
            'out_channels': 64,
            'use_layernorm': True,
            'final_norm': 'layernorm',
            'normal_channel': False,
        })

        self.model = DP3(
            use_point_crop=True,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,

            diffusion_step_embed_dim=128,
            down_dims=(512, 1024, 2048),
            crop_shape=(80, 80),
            encoder_output_dim=64,
            horizon=8,
            kernel_size=5,
            n_action_steps=8,
            n_groups=8,
            n_obs_steps=2,
            noise_scheduler=noise_scheduler,
            num_inference_steps=10,
            obs_as_global_cond=True,
            shape_meta=shape_meta,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=pointcloud_endoer_cfg,
        )

        normalizer = LinearNormalizer()

        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
        self.model.set_normalizer(normalizer)

    def forward(self, batch):
        return self.model.compute_loss(batch)

    def inference_one_sample(self, batch):
        self.model.predict_action(batch['obs'])


if __name__ == "__main__":
    dp3 = DP3Wrapper()

    batch = {
        'obs': {
            'point_cloud': torch.randn(2, 10, 4096, 3),
            'agent_pos': torch.randn(2, 10, 8),
        },
        'action': torch.randn(2, 8, 8),
    }

    loss = dp3(batch)
    print(loss)
