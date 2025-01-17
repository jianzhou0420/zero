import open3d as o3d
import pickle
import numpy as np
import torch
file_path = '/media/jian/ssd4t/selfgen/test/test/seed42/test1111/close_jar/variation0/episodes/episode0/data.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

for t in range(len(data['pc_fts'])):
    pc_fts = data['pc_fts'][t]

    xyz = np.array(pc_fts[:, :3])
    rgb = np.array((pc_fts[:, 3:6] + 1) / 2)

    print('pc_fts', pc_fts)
    print('pc_centroids', data['pc_centroids'][t])
    print('pc_radius', data['pc_radius'][t])
    print('current', torch.round(data['ee_poses'][t], decimals=2))
    print('next111', torch.round(data['gt_actions'][t], decimals=2))

    data['ee_poses'][t][:3] = data['ee_poses'][t][:3] + data['pc_centroids'][t]
    data['gt_actions'][t][:3] = data['gt_actions'][t][:3] + data['pc_centroids'][t]

    print('current', torch.round(data['ee_poses'][t], decimals=2))
    print('next111', torch.round(data['gt_actions'][t], decimals=2))

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5,  # Size of the axes
        origin=[0, 0, 0]  # Origin of the axes
    )

    print('sum', np.round(sum(xyz, 0)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([pcd, coordinate_frame])
