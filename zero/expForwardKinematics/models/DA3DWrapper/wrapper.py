import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor


class Wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从bash /data/zero/wrapper/3d_diffuser_actor/scripts/train_keypose_peract.sh中获得。
        backbone = 'clip'
        image_size = (256, 256)
        embedding_dim = 120
        num_vis_ins_attn_layers = 2
        use_instruction = True
        fps_subsampling_factor = 5
        gripper_loc_bounds = [[-0.06862151, -0.5557904, 0.72967406], [0.64813266, 0.44632257, 1.43157236]]
        rotation_parametrization = '6D'
        quaternion_format = 'xyzw'
        diffusion_timesteps = 100
        nhist = 3
        relative = False
        lang_enhanced = False

        self.model = DiffuserActor(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            use_instruction=use_instruction,
            fps_subsampling_factor=fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=rotation_parametrization,
            quaternion_format=quaternion_format,
            diffusion_timesteps=diffusion_timesteps,
            nhist=nhist,
            relative=relative,
            lang_enhanced=lang_enhanced
        )

    def forward(self, batch, *args, **kwargs):

        return self.model(*args, **kwargs)
