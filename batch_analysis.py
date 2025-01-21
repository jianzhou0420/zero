import pickle
import torch
import numpy as np
import open3d as o3d
with open('data_mine.pickle', 'rb') as f:
    data_mine = pickle.load(f)

with open('data_origin.pickle', 'rb') as f:
    data_origin = pickle.load(f)


# for i in range(500):
#     pc_fts_mine = data_mine['pc_fts'][i]
#     pc_fts_origin = data_origin['pc_fts'][i]

#     xyz_mine = pc_fts_mine[:, :3]
#     rgb_mine = (pc_fts_mine[:, 3:6] + 1) / 2

#     action_next_mine = data_mine['gt_actions'][i]
#     action_next_xyz_mine = action_next_mine[:3].unsqueeze(0)

#     rgb_action = np.zeros((1, 3))
#     rgb_action[0] = [1, 0, 0]

#     xyz = np.concatenate((xyz_mine, action_next_xyz_mine), axis=0)
#     rgb = np.concatenate((rgb_mine, rgb_action), axis=0)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd.colors = o3d.utility.Vector3dVector(rgb)
#     o3d.visualization.draw_geometries([pcd])


# for i in range(10):
#     # pc_fts_mine = data_mine['pc_fts'][i]
#     pc_fts_mine = data_origin['pc_fts'][i]

#     xyz_mine = pc_fts_mine[:, :3]
#     rgb_mine = (pc_fts_mine[:, 3:6] + 1) / 2

#     action_next_mine = data_origin['gt_actions'][i]
#     action_next_xyz_mine = action_next_mine[:3].unsqueeze(0)

#     rgb_action = np.zeros((1, 3))
#     rgb_action[0] = [1, 0, 0]

#     xyz = np.concatenate((xyz_mine, action_next_xyz_mine), axis=0)
#     rgb = np.concatenate((rgb_mine, rgb_action), axis=0)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     pcd.colors = o3d.utility.Vector3dVector(rgb)
#     o3d.visualization.draw_geometries([pcd])

#     # action_xyz = data_mine[i]['gt_actions'][:, :3]

# region centroid and radius
# import numpy as np
# import csv
# centroids_mine_list = []
# radius_mine_list = []
# for item in data_mine['pc_centroids']:
#     centroids_mine_list.append(item)

# for item in data_mine['pc_radius']:
#     radius_mine_list.append(item)

# centroids_mine = np.array(centroids_mine_list)
# radius_mine = np.array(radius_mine_list)

# with open('centroids_mine.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(centroids_mine)


# centroids_origin_list = []
# radius_origin_list = []
# for item in data_origin['pc_centroids']:
#     centroids_origin_list.append(item)

# for item in data_origin['pc_radius']:
#     radius_origin_list.append(item)

# centroids_origin = np.array(centroids_origin_list)
# radius_origin = np.array(radius_origin_list)

# with open('centroids_origin.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows(centroids_origin)
# endregion

# region pc_numbers

# import numpy as np
# import csv
# pc_fts_mine_list = []
# pc_fts_origin_list = []

# for item in data_mine['pc_fts']:
#     pc_fts_mine_list.append(len(item))

# for item in data_origin['pc_fts']:
#     pc_fts_origin_list.append(len(item))

# pc_fts_mine = np.array(pc_fts_mine_list)
# pc_fts_origin = np.array(pc_fts_origin_list)

# with open('pc_numbers_mine.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(pc_fts_mine)

# with open('pc_numbers_origin.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(pc_fts_origin)
# # endregion


with open('data.pkl', 'rb') as f:
    data = pickle.load(f)


for i in range(10):
    data_single = data[i]
    for j in range(len(data_single['pc_fts'])):
        pc_ft = data_single['pc_fts'][j]
        xyz = pc_ft[:, :3]
        rgb = (pc_ft[:, 3:6] + 1) / 2

        action_next = data_single['gt_actions'][j]
        xyz_action = action_next[:3].unsqueeze(0)
        rgb_action = np.zeros((1, 3))

        rgb_action[0] = [1, 0, 0]
        xyz = np.concatenate((xyz, xyz_action), axis=0)
        rgb = np.concatenate((rgb, rgb_action), axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.visualization.draw_geometries([pcd])
