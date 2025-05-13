from zero.expForwardKinematics.config.default import get_config
from einops import rearrange
from codebase.z_utils.Rotation import *
from codebase.z_utils.rotation_import import *
import pickle
import open3d as o3d
from zero.expForwardKinematics.ReconLoss.FrankaPandaFK import FrankaEmikaPanda
from zero.expForwardKinematics.ObsProcessor.ObsProcessorFKAll import ObsProcessorFK
from numpy.linalg import inv as matinv
"""
theta_the means theta_theoretical,
theta_sim means theta_simulated

"""
config_path = './zero/expForwardKinematics/config/DP_traj.yaml'
config = get_config(config_path)

obs_processor = ObsProcessorFK(config)


def get_robot_pcd_idx(xyz, obbox):
    points = o3d.utility.Vector3dVector(xyz)
    robot_point_idx = set()
    for box in obbox:
        tmp = box.get_point_indices_within_bounding_box(points)
        robot_point_idx = robot_point_idx.union(set(tmp))
    robot_point_idx = np.array(list(robot_point_idx))
    mask = np.zeros(len(xyz), dtype=bool)
    mask[robot_point_idx] = True
    return mask


franka = FrankaEmikaPanda()


episode_path = './1_Data/B_Preprocess/DA3D/train/put_groceries_in_cupboard/variation0/episodes/episode0/data.pkl'

frame_id = 1
with open(episode_path, 'rb') as f:
    episode = pickle.load(f)

# print(episode.keys())
theta_offset = npa([0, 0, 0, radians(-4), 0, 0, 0])

theta_sim = episode['JP_hist'][frame_id][-1][:-1]
theta_the = theta_sim - franka.JP_offset[:-1]

rgb = episode['rgb'][0]
xyz = episode['xyz'][0]


xyz = rearrange(xyz, 'ncam h w c -> (ncam h w) c')
rgb = rearrange(rgb, 'ncam h w c -> (ncam h w) c')

xyz, rgb = obs_processor.within_workspace(xyz, rgb)
xyz, rgb = obs_processor.remove_table(xyz, rgb)
xyz, rgb = obs_processor.voxelize(xyz, rgb)


theta_the = np.hstack([theta_the, 0])
bbox_link, bbox_other = franka.theta2obbox(theta_the)


# P_k = T_ok @ P_o.T
bbox_all = bbox_link + bbox_other
pcd_idx = get_robot_pcd_idx(xyz, bbox_all)
xyz = xyz[~pcd_idx]
rgb = rgb[~pcd_idx]


# show eePose_gt
eePose = episode['eePose_hist'][frame_id][-1]
eePosePos = eePose[:3][None, :]
# print(eePosePos)
xyz = np.concatenate([xyz, eePosePos], axis=0)
rgb = np.concatenate([rgb, np.array([[255, 0, 0]])], axis=0)


# get eePose_calculated
_, T_oi = franka.get_T_oi(theta_the)

T_oi_last = T_oi[-1]

T_eePose = eePose2HT(eePose[:-1])

T_last2eePose = matinv(T_oi_last) @ T_eePose

print('T_last2eePose', T_last2eePose)


# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
# # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# o3d.visualization.draw_geometries([pcd, *bbox_all])
# o3d.visualization.draw_geometries([cl,])
