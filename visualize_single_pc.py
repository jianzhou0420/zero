import open3d as o3d
import pickle

file_path = '/media/jian/ssd4t/selfgen/seed42/voxel0.005/close_jar/variation0/episodes/episode0/data.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

xyz = data['pc'][0]
rgb = data['rgb'][0]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

o3d.visualization.draw_geometries([pcd])
