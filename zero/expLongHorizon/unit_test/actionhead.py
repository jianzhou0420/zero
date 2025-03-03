import einops
import torch
from zero.expLongHorizon.models.lotus.long_horizon_head import MultiActionHead
from zero.expLongHorizon.models.lotus.model_expbase import ActionHead

from zero.expLongHorizon.config.default import get_config


config_path = '/data/zero/zero/expLongHorizon/config/expBase_Lotus.yaml'


parameters = [
    "name", "EXP03_02_rollback",
    "dataset", "augment",
    "num_gpus", "1",
    "epoches", "400",
    "batch_size", "4",
    "TRAIN_DATASET.num_points", "100000",
    "TRAIN_DATASET.pos_bins", "75",
    "TRAIN_DATASET.pos_bin_size", "0.001",
    "MODEL.action_config.pos_bins", "75",
    "MODEL.action_config.pos_bin_size", "0.001",
    "tasks_to_use", "$tasks_to_use",
    "horizon", "8",
    "TRAIN_DATASET.pos_heatmap_no_robot", "False",
    "MODEL.action_config.horizon", "8",
    "MODEL.action_config.action_head_type", "multihead",
    "unit_test", "True",  # attention! unit_test!
]


config = get_config(config_path, parameters)

act_cfg = config.MODEL.action_config
config = config.MODEL

singlehead = ActionHead(act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
                        config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
                        dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
                        ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
                        euler_resolution=config.action_config.euler_resolution,
                        unit_test=True)  # unit_test=True!


multihead = MultiActionHead(act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
                            config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
                            dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
                            ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
                            euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon,
                            unit_test=True)  # unit_test=True!


multihead.heatmap_mlp_list[0].load_state_dict(singlehead.heatmap_mlp.state_dict())
multihead.action_mlp_list[0].load_state_dict(singlehead.action_mlp.state_dict())

sample1 = torch.ones(1, 128)
sample2 = torch.ones(1, 128)
npoints_in_batch = 1


xt1, xr1, xo1 = singlehead(sample1, npoints_in_batch)

xt2, xr2, xo2 = multihead(sample2, npoints_in_batch)

weight1 = singlehead.heatmap_mlp.state_dict()
weight2 = multihead.heatmap_mlp_list[0].state_dict()
for key in weight1:
    # Using torch.allclose allows for some floating-point tolerance.
    if not torch.allclose(weight1[key], weight2[key]):
        print('heatmap_mlp weight is different')
        break
    print('heatmap_mlp weight is same')

weight1 = singlehead.action_mlp.state_dict()
weight2 = multihead.action_mlp_list[0].state_dict()

for key in weight1:
    # Using torch.allclose allows for some floating-point tolerance.
    if not torch.allclose(weight1[key], weight2[key]):
        print('action_mlp weight is different')
        break
    print('action_mlp weight is same')

xt1 = einops.rearrange(xt1, 'c b d -> b c d')

print(xt1.shape, xr1.shape, xo1.shape)
print(xt2.shape, xr2.shape, xo2.shape)


test1 = xr1
test2 = xr2[:, 0, :, :]
print('xt is', (xt1 == xt2[:, 0, :, :]).all())
print('xr is', (xr1 == xr2[:, 0, :, :]).all())
print('xo is', (xo1 == xo2[0, 0, 0]).all())
