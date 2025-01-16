import open3d as o3d
import pickle
import numpy as np
file_path = '/media/jian/ssd4t/selfgen/20250115/seed42/voxel0.005/close_jar/variation0/episodes/episode0/data.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
pc_fts = data['pc_fts'][0]

xyz = np.array(pc_fts[:, :3])
rgb = np.array((pc_fts[:, 3:6] + 1) / 2)


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(rgb)

# o3d.visualization.draw_geometries([pcd])


print(data['pc_centroids'][0])
print(data['pc_radius'][0])
