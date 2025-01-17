from zero.v2.models.lotus.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
import numpy as np
from zero.v2.dataprocess.ObsProcessLotus import ObsProcessLotus
import yaml
import pickle
import yacs.config
from zero.v1.models.lotus.utils.robot_box import RobotBox
config = yacs.config.CfgNode(new_allowed=True)
config.merge_from_file('/workspace/zero/zero/v2/config/lotus_exp2_0.005_close_jar.yaml')

example_data_file = '/media/jian/ssd4t/selfgen/20250105/train_dataset/keysteps/seed42/close_jar/variation0/episodes/episode0/data.pkl'
with open(example_data_file, 'rb') as f:
    data = pickle.load(f)

voxel_size = 0.005
op = ObsProcessLotus(config.TRAIN_DATASET, voxel_size)

print(data.keys())  # dict_keys(['key_frameids', 'rgb', 'pc', 'action', 'gripper_pose_heatmap', 'bbox', 'pose']) # gripper_pos_heatmap is useless

t = 0
xyz = data['pc'][t]
rgb = data['rgb'][t]
action_current = data['action'][t]
action_next = data['action'][t + 1]
bbox = data['bbox'][t]
pose = data['pose'][t]

arm_links_info = (bbox, pose)
is_train = True
batch = {
    'data_ids': [],
    'pc_fts': [],
    'step_ids': [],
    'pc_centroids': [],
    'pc_radius': [],
    'ee_poses': [],
    'txt_embeds': [],
    'gt_actions': [],
    'disc_pos_probs': [],
}
# single_frame process start, target to get batch

xyz, rgb = op.process_pc(xyz, rgb, arm_links_info, voxel_size)  # only do remove


robot_box = RobotBox(arm_links_info=arm_links_info, env_name='rlbench',)
robot_point_idxs = np.array(list(robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1]))  # 需要放在pc缩减之后，augment之前
height = xyz[:, -1] - 0.7505  # 相当于每个点对于桌面的高度，其实我觉得应该放在augment之后，不过先按照作者的思路来。

if is_train:
    angle = np.random.uniform(-1, 1) * op.config.aug_max_rot
    xyz = op.augment_xyz(xyz, angle)
    action_current = op.augment_action(action_current, angle)
    action_next = op.augment_action(action_next, angle)


# normalize
centroid = np.mean(xyz, axis=0)
radius = 1
xyz = (xyz - centroid) / radius
action_current[:3] = (action_current[:3] - centroid) / radius

if is_train:
    action_next[:3] = (action_next[:3] - centroid) / radius

# post-process
# 1.convert action_next's quaternion to discrete euler
    action_next_rot = quaternion_to_discrete_euler(action_next[3:-1], op.config.euler_resolution)

# 2.get gt_pos_prob 是最后再做的，约等于转换以下action_next
    disc_pos_prob = op.action_next_pos_prob(
        xyz, action_next[:3], pos_bins=op.config.pos_bins,
        pos_bin_size=op.config.pos_bin_size,
        heatmap_type=op.config.pos_heatmap_type,
        robot_point_idxs=robot_point_idxs
    )
    batch['disc_pos_probs'].append(disc_pos_prob)
pt_ft = np.concatenate((xyz, rgb, height[:, None]), axis=1)
gt_action = np.concatenate([action_next[:3], action_next_rot, action_next[-1:]], axis=0)

pt_ft, action_current, action_next, centroid, radius

# collect for batch


print
