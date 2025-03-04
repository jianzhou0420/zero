import open3d as o3d
import pickle
import numpy as np
import torch
file_path = '/data/selfgen/after_shock3/close_jar/variation4/episodes/episode0/data.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

for t in range(len(data['xyz'])):

    xyz = data['xyz'][t]
    # rgb = np.array((pc_fts[:, 3:6] + 1) / 2)
    rgb = data['rgb'][t]

    ee_pose = data['action_current'][t][:3]
    ee_rgb = np.array([255, 0, 0])

    gt_action = data['action_next'][t][:3]
    gt_rgb = np.array([0, 255, 0])

    xyz = np.array(xyz)
    rgb = np.array(rgb)
    xyz = np.vstack([xyz, ee_pose, gt_action])
    rgb = np.vstack([rgb, ee_rgb, gt_rgb])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5,  # Size of the axes
        origin=[0, 0, 0]  # Origin of the axes
    )

    print('sum', np.round(sum(xyz, 0)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255)

    o3d.visualization.draw_geometries([pcd, coordinate_frame])
