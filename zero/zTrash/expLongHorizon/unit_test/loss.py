from torch.utils.data import DataLoader
import open3d as o3d
import yacs.config
from ..dataset.dataset_expbase_voxel_augment_with_path import LotusDatasetAugmentation as augmentation_dataset_with_path
from ..dataset.dataset_expbase_voxel_augment import LotusDatasetAugmentation as augmentation_dataset
from ..dataset.dataset_expbase_voxel_augment import ptv3_collate_fn
import yaml
import os
import numpy as np
import einops
import torch
from ..models.lotus.long_horizon_head import SimplePolicyPTV3CA as Multipolicy
from ..models.lotus.model_expbase import SimplePolicyPTV3CA as Singlepolicy

from ..config.default import get_config


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
    "unit_test", "True",
    "horizon", "8",
    "TRAIN_DATASET.pos_heatmap_no_robot", "False",
    "MODEL.action_config.horizon", "8",
    "MODEL.action_config.action_head_type", "multihead",
]

# first get verified dataset pair


tasks_to_use = 'close_jar'

B_PREPROCESS_FACTORY = {
    '0.01all_with_path': '/data/zero/1_Data/B_Preprocess/0.01all_with_path',
    '0.005all': '/data/zero/1_Data/B_Preprocess/0.005all',
    '0.005all_with_path': '/data/zero/1_Data/B_Preprocess/0.005all_with_path',
}


config = get_config(config_path, parameters)
# print(config)
train_data_dir1 = os.path.join(B_PREPROCESS_FACTORY['0.005all_with_path'], 'train')
train_data_dir2 = os.path.join(B_PREPROCESS_FACTORY['0.005all_with_path'], 'train')

dataset1 = augmentation_dataset(config=config, tasks_to_use=tasks_to_use, data_dir=train_data_dir1, **config.TRAIN_DATASET)
dataset2 = augmentation_dataset_with_path(config=config, tasks_to_use=tasks_to_use, data_dir=train_data_dir2, **config.TRAIN_DATASET)


# then get verified model pair

singlepolicy = Singlepolicy(config.MODEL, unit_test=True).to('cuda')
multipolicy = Multipolicy(config.MODEL, unit_test=True).to('cuda')
multipolicy.act_proj_head.heatmap_mlp_list[0].load_state_dict(singlepolicy.act_proj_head.heatmap_mlp.state_dict())
multipolicy.act_proj_head.action_mlp_list[0].load_state_dict(singlepolicy.act_proj_head.action_mlp.state_dict())

weight1 = singlepolicy.act_proj_head.heatmap_mlp.state_dict()
weight2 = multipolicy.act_proj_head.heatmap_mlp_list[0].state_dict()
for key in weight1:
    # Using torch.allclose allows for some floating-point tolerance.
    if not torch.allclose(weight1[key], weight2[key]):
        print('heatmap_mlp weight is different')
        break
    print('heatmap_mlp weight is same')

weight1 = singlepolicy.act_proj_head.action_mlp.state_dict()
weight2 = multipolicy.act_proj_head.action_mlp_list[0].state_dict()

for key in weight1:
    # Using torch.allclose allows for some floating-point tolerance.
    if not torch.allclose(weight1[key], weight2[key]):
        print('action_mlp weight is different')
        break
    print('action_mlp weight is same')


dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False, collate_fn=ptv3_collate_fn)
dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False, collate_fn=ptv3_collate_fn)


single_batch_1 = next(iter(dataloader1))
single_batch_2 = next(iter(dataloader2))
# (['data_ids', 'pc_fts', 'step_ids', 'pc_centroids', 'pc_radius', 'ee_poses', 'txt_embeds', 'gt_actions', 'disc_pos_probs', 'npoints_in_batch', 'offset', 'txt_lens'])


flag1 = (single_batch_1['data_ids'] == single_batch_2['data_ids'])
flag2 = (single_batch_1['pc_fts'] == single_batch_2['pc_fts']).all()
flag3 = (single_batch_1['step_ids'] == single_batch_2['step_ids']).all()
flag4 = (single_batch_1['pc_centroids'] == single_batch_2['pc_centroids']).all()
flag5 = (single_batch_1['pc_radius'] == single_batch_2['pc_radius'])
flag6 = (single_batch_1['ee_poses'] == single_batch_2['ee_poses']).all()
# flag7 = (single_batch_1['txt_embeds'] == single_batch_2['txt_embeds']).all()
flag8 = (single_batch_1['gt_actions'] == single_batch_2['gt_actions'][:, -1, :]).all()
flag9 = (single_batch_1['disc_pos_probs'][0] == single_batch_2['disc_pos_probs'][0][-1]).all()
flag10 = (single_batch_1['npoints_in_batch'] == single_batch_2['npoints_in_batch'])
flag11 = (single_batch_1['offset'] == single_batch_2['offset']).all()
# flag12 = (single_batch_1['txt_lens'] == single_batch_2['txt_lens'])

print('data_ids', flag1)
print('pc_fts', flag2)
print('step_ids', flag3)
print('pc_centroids', flag4)
print('pc_radius', flag5)
print('ee_poses', flag6)
# print('txt_embeds', flag7)
print('gt_actions', flag8)
print('disc_pos_probs', flag9)
print('npoints_in_batch', flag10)
print('offset', flag11)
# print('txt_lens', flag12)
print('all is good', flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag8 and flag9 and flag10 and flag11)

print('#' * 100)
print('#' * 100)
print('#' * 100)


for i, item in enumerate(single_batch_1['disc_pos_probs']):
    single_batch_1['disc_pos_probs'][i] = item.to('cuda')


for i, item in enumerate(single_batch_2['disc_pos_probs']):
    for j, horizon in enumerate(item):
        single_batch_2['disc_pos_probs'][i][j] = horizon.to('cuda')

torch.manual_seed(42)
xt1, xr1, xo1, xt_input1, xt_target1, xr_input1, xr_target1, xo_input1, xo_target1 = singlepolicy(single_batch_1, is_train=True)
torch.manual_seed(42)
xt2, xr2, xo2, xt_input2, xt_target2, xr_input2, xr_target2, xo_input2, xo_target2 = multipolicy(single_batch_2, is_train=True)

xt1 = einops.rearrange(xt1, 'c n b -> n c b')
atest1 = xr1
atest2 = xr2[:, 0, :, :]
print('xt is', (xt1 == xt2[:, 0, :, :]).all())
print('xr is', (xr1 == xr2[:, 0, :, :]).all())
print('xo is', (xo1 == xo2[0, 0, 0]).all())

print('xt', torch.allclose(xt1, xt2))
