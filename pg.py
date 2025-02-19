
import open3d as o3d
import pickle
import numpy as np
data_path = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/0.005all/train/close_jar/variation0/episodes/episode0/data.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
xyz = data['xyz'][0]
rgb = data['rgb'][0]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector((rgb / 255).astype(np.float64))
o3d.visualization.draw_geometries([pcd])
