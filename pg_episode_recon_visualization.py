from zero.expAugmentation.config.default import get_config
from einops import rearrange
from codebase.z_utils.Rotation import *
from codebase.z_utils.rotation_import import *
import pickle
import open3d as o3d
from zero.expAugmentation.ReconLoss.ForwardKinematics import FrankaEmikaPanda
from zero.expAugmentation.ObsProcessor.ObsProcessorPtv3 import ObsProcessorPtv3
"""
theta_the means theta_theoretical,
theta_sim means theta_simulated

"""
config_path = '/media/jian/ssd4t/zero/zero/expAugmentation/config/DP.yaml'
config = get_config(config_path)

obs_processor = ObsProcessorPtv3(config)


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


episode_path = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DA3D/close_jar/variation0/episodes/episode0/data.pkl'


with open(episode_path, 'rb') as f:
    episode = pickle.load(f)

print(episode.keys())
theta_offset = npa([0, 0, 0, radians(-4), 0, 0, 0])

theta_sim = episode['joint_position_history'][0][-1][:-1]  # 第一帧的关节角度
theta_the = theta_sim - theta_offset

rgb = episode['rgb'][0]
xyz = episode['pcd'][0]


xyz = rearrange(xyz, 'ncam h w c -> (ncam h w) c')
rgb = rearrange(rgb, 'ncam h w c -> (ncam h w) c')

xyz, rgb = obs_processor.within_workspace(xyz, rgb)
xyz, rgb = obs_processor.remove_table(xyz, rgb)
xyz, rgb = obs_processor.voxelize(xyz, rgb)


theta_the = np.hstack([theta_the, 0])
bbox_link, bbox_other = franka.theta2obbox(theta_the)


# ones = np.ones((len(xyz), 1))
# P_o = np.hstack((xyz, ones))


# P_k = T_ok @ P_o.T

bbox_all = bbox_link + bbox_other
pcd_idx = get_robot_pcd_idx(xyz, bbox_all)
xyz = xyz[~pcd_idx]
rgb = rgb[~pcd_idx]


# print(P_k.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

o3d.visualization.draw_geometries([cl, *bbox_all])
