import open3d as o3d
import numpy as np
import cv2
import pickle
from zero.v3.dataprocess.ObsProcessor import ObsProcessLotus
import yacs.config
import matplotlib.pyplot as plt
'''
# outs = op.dataset_preprocess_single_episode(data=data, episode_path=single_episode_path)
# outs = op.dataset_dataloader_process(outs)
# print(outs.keys())


# 1. remove points outside the bounding box
# 2. remove table points
# 3. voxelization
# -----
# 4. remove robots
# 5. Downsampling
# 6. augment
# 7. normalize


# (1) 学习的是不同位置，不同颜色，不同物体关系的泛化性。
# (2) augment确保这一点。


version 1:


# for each frame select instruction embedding
'''
import copy

single_episode_path = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/seed42/place_shape_in_shape_sorter/variation0/episodes/episode0/data.pkl'
config_path = '/media/jian/ssd4t/zero/zero/v3/config/sort_shape_edge.yaml'
config = yacs.config.CfgNode(new_allowed=True)
config.merge_from_file(config_path)

op = ObsProcessLotus(config=config)


with open(single_episode_path, 'rb') as f:
    data = pickle.load(f)  # dict_keys(['key_frameids', 'rgb', 'pc', 'action', 'gripper_pose_heatmap', 'bbox', 'pose'])

for i in range(len(data['rgb'])):
    first_frame_xyz = copy.deepcopy(data['pc'][i])
    first_frame_rgb = copy.deepcopy(data['rgb'][i])

    idxs = []

    for image in first_frame_rgb[:]:
        canny_edges = cv2.Canny(image, 150, 200)
        idxs.append(canny_edges > 0)

    all_points = []
    all_points_rgb = []
    for i, idx in enumerate(idxs):
        single_image_points = first_frame_xyz[i][idx]
        single_image_points_rgb = first_frame_rgb[i][idx]
        all_points_rgb.append(single_image_points_rgb)
        all_points.append(single_image_points)

    xyz = np.vstack(all_points)
    rgb = np.vstack(all_points_rgb)
    ##########################################

    xyz, rgb = op.inside_workspace(xyz, rgb)
    xyz, rgb = op.remove_table(xyz, rgb)

    print('inside workspace:', xyz.shape)

    ##########################################

    op.visualize_pc(xyz, rgb / 255)
