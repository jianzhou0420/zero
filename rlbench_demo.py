import copy
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget
import pyrep.objects.joint
from zero.expAugmentation.ObsProcessor.ObsProcessorPtv3 import ObsProcessorPtv3
from zero.expAugmentation.config.default import get_config
import open3d as o3d
from zero.expAugmentation.models.lotus.utils.robot_box import RobotBox
# some args
LINK_NUM = 8

headless = 1
JOINT_POSITIONS_LIMITS = np.array([[-2.8973, 2.8973],
                                   [-1.7628, 1.7628],
                                   [-2.8973, 2.8973],
                                   [-3.0718, -0.0698],
                                   [-2.8973, 2.8973],
                                   [-0.0175, 3.7525],
                                   [-2.8973, 2.8973]])

# ptv3 obs processor
config_path = "/data/zero/zero/expAugmentation/config/DP.yaml"
config = get_config(config_path)
config.defrost()
config.TRAIN_DATASET.rm_robot = False
config.freeze()
obs_processor = ObsProcessorPtv3(config, train_flag=0)

# end of ptv3 obs processor

# main


def get_mask_with_obb(self, xyz, arm_links_info, rm_robot_type):
    if rm_robot_type == 'box_keep_gripper':
        keep_gripper = True
    else:
        keep_gripper = False

    robot_box = RobotBox(
        arm_links_info, keep_gripper=keep_gripper,
        env_name='rlbench', selfgen=True
    )
    _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
    robot_point_ids = np.array(list(robot_point_ids))
    mask = np.ones((xyz.shape[0], ), dtype=bool)
    if len(robot_point_ids) > 0:
        mask[robot_point_ids] = False

    return mask


def playground(obs, link_pose_theta_all_0, joints_locations, joints_position):
    # 仅仅代码结构，用的都是global变量
    obs_raw = obs_processor.obs_2_obs_raw(obs)
    static_process = obs_processor.static_process_fk(obs_raw)

    # link pose
    xyz = copy.deepcopy(static_process['xyz'][0])
    rgb = copy.deepcopy(static_process['rgb'][0])
    link_pose_point = link_pose_theta_all_0[:, :3]

    # print(static_process)
    xyz = np.vstack([link_pose_point, xyz])
    link_pose_rgb = np.zeros((LINK_NUM, 3))
    link_pose_rgb[:, 0] = 255
    rgb = np.vstack([link_pose_rgb, rgb])

    # joint pose
    jl = np.array(joints_locations)
    xyz = np.vstack([jl, xyz])
    joint_pose_rgb = np.zeros((7, 3))
    joint_pose_rgb[:, 1] = 255
    rgb = np.vstack([joint_pose_rgb, rgb])

    # obb
    arm_links_info = (obs_raw['bbox'][0], obs_raw['pose'][0])
    robot_box = RobotBox(
        arm_links_info, keep_gripper=True,
        env_name='rlbench', selfgen=True
    )
    obb = robot_box.robot_obboxes
    for single_obb in obb:
        single_obb.color = (0, 0, 1)

    for item in joints_locations:
        item -= [-0.26796156, -0.00639182, 0.7505]
    # visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    o3d.visualization.draw_geometries([pcd, *obb])
    pass


def get_Frames_Transformation():
    from math import cos, sin
    import math

    def dh_modified_transform(alpha, a, theta, d):
        ct = cos(theta)
        st = sin(theta)

        ca = cos(alpha)
        sa = sin(alpha)

        t11 = ct
        t12 = -st
        t13 = 0
        t14 = a

        t21 = st * ca
        t22 = ct * ca
        t23 = -sa
        t24 = -d * sa

        t31 = st * sa
        t32 = ct * sa
        t33 = ca
        t34 = d * ca

        t41 = 0
        t42 = 0
        t43 = 0
        t44 = 1

        T = np.array([[t11, t12, t13, t14],
                      [t21, t22, t23, t24],
                      [t31, t32, t33, t34],
                      [t41, t42, t43, t44]])
        return T

    theta = [0, 0, 0, -0.0698, 0, 0, 0]
    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, ]
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, ]
    alpha = [0, -math.pi / 2, math.pi / 2, math.pi / 2,
             -math.pi / 2, math.pi / 2, math.pi / 2, ]

    T_i_1_i_list = []
    for i in range(7):
        T = dh_modified_transform(alpha[i], a[i], theta[i], d[i])
        T_i_1_i_list.append(T)

    joint_positions = []
    T_cumulative = np.eye(4)
    T_i_list = []
    for T in T_i_1_i_list:
        T_cumulative = T_cumulative @ T
        T_i_list.append(copy.deepcopy(T_cumulative))

        # pos = T_cumulative[:3, 3]
        # joint_positions.append(pos)

    return T_i_1_i_list, T_i_list

# end of main


def main():
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointPosition(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode, headless=headless)
    env.launch()

    task = env.get_task(ReachTarget)
    descriptions, obs = task.reset()
    for i in range(1000):
        action = np.zeros(env.action_shape)
        # action = np.random.randn(*action.shape)
        obs, reward, terminate = task.step(action)

        # 1. get the position of links, but I dont know which is the part of the robot
        link_pose_theta_all_0 = []
        for link_id in range(LINK_NUM):
            link_pose_theta_all_0.append(obs.misc[f'Panda_link{link_id}_visual_pose'])
            # break
        link_pose_theta_all_0 = np.array(link_pose_theta_all_0)
        # print(link_pose_theta_all_0)
        # 1.2 get joint poses
        joints_locations = [env._robot.arm.joints[i].get_position() for i in range(7)]
        joints_position = [env._robot.arm.joints[i].get_joint_position() for i in range(7)]

        # 2.from env to get the robot info
        if i >= 50:
            # test(obs, link_pose_theta_all_0, joints_locations, joints_position)
            calculate_T_JL(obs, link_pose_theta_all_0, joints_locations, joints_position)
            pass
        else:
            # test(obs, link_pose_theta_all_0, joints_locations)
            pass
    T_prev_to_curr, T_zero_to_curr = get_Frames_Transformation()
    T_Joints_theory = np.array(T_zero_to_curr)


def calculate_T_JL(obs, link_pose_theta_all_0, joints_locations, joints_position):

    print("calculate_T_JL")

    pass


main()
