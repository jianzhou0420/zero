import numpy as np
import os
from ..config.default import get_config
import yaml
from ..dataset.dataset_expbase_voxel_augment import LotusDatasetAugmentation as augmentation_dataset
from ..dataset.dataset_expbase_voxel_augment_with_path import LotusDatasetAugmentation as augmentation_dataset_with_path
import yacs.config


import open3d as o3d
tasks_to_use = 'put_groceries_in_cupboard'

B_PREPROCESS_FACTORY = {
    '0.01all_with_path': '/data/zero/1_Data/B_Preprocess/0.01all_with_path',
    '0.005all': '/data/zero/1_Data/B_Preprocess/0.005all',
    '0.005all_with_path': '/data/zero/1_Data/B_Preprocess/0.005all_with_path',
}


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
    "horizon", "10",
    "TRAIN_DATASET.pos_heatmap_no_robot", "False",

]

config = get_config(config_path, parameters)
# print(config)
train_data_dir1 = os.path.join(B_PREPROCESS_FACTORY['0.005all_with_path'], 'train')
train_data_dir2 = os.path.join(B_PREPROCESS_FACTORY['0.005all_with_path'], 'train')

dataset1 = augmentation_dataset(config=config, tasks_to_use=tasks_to_use, data_dir=train_data_dir1, **config.TRAIN_DATASET)
dataset2 = augmentation_dataset_with_path(config=config, tasks_to_use=tasks_to_use, data_dir=train_data_dir2, **config.TRAIN_DATASET)

print(dataset1[0].keys())
print(dataset2[0].keys())
# dict_keys(['data_ids', 'pc_fts', 'step_ids', 'pc_centroids', 'pc_radius', 'ee_poses', 'txt_embeds', 'gt_actions', 'disc_pos_probs'])

# print('data_ids', [dataset1[0]['data_ids'][i] == dataset2[0]['data_ids'][i] for i in range(len(dataset2[0]['data_ids']))])
# print('pc_fts', [dataset1[0]['pc_fts'][i] == dataset1[0]['pc_fts'][i] for i in range(len(dataset2[0]['pc_fts']))])
# print('step_ids', [dataset1[0]['step_ids'][i] == dataset2[0]['step_ids'][i] for i in range(len(dataset2[0]['step_ids']))])
# print('pc_centroids', [dataset1[0]['pc_centroids'][i] == dataset2[0]['pc_centroids'][i] for i in range(len(dataset2[0]['pc_centroids']))])
# print('pc_radius', [dataset1[0]['pc_radius'][i] == dataset2[0]['pc_radius'][i] for i in range(len(dataset2[0]['pc_radius']))])
# print('ee_poses', [dataset1[0]['ee_poses'][i] == dataset2[0]['ee_poses'][i] for i in range(len(dataset2[0]['ee_poses']))])


gt_action1 = dataset1[0]['gt_actions'][0]
gt_action2 = dataset2[0]['gt_actions'][0]
print('gt_actions', gt_action1 == gt_action2[-1])

demo_num = 10
for i in range(demo_num):

    random_i = np.random.randint(len(dataset1))
    for j in range(len(dataset2[random_i]['pc_fts'])):
        pc_fts = dataset2[random_i]['pc_fts'][j].numpy()
        xyz = pc_fts[:, :3]
        rgb = (pc_fts[:, 3:6] + 1) / 2

        action_xyz = dataset2[random_i]['gt_actions'][j].numpy()[:, :3]
        action_rgb = np.zeros((len(action_xyz), 3))

        xyz = np.vstack([xyz, action_xyz],)
        rgb = np.vstack([rgb, action_rgb])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])

        # visualize gt_actions
