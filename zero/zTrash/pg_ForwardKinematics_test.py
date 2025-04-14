
import math
import copy
import numpy as np
import copy
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget
import pyrep.objects.joint
from zero.expForwardKinematics.ObsProcessor.ObsProcessorPtv3_fk import ObsProcessorPtv3
from zero.expForwardKinematics.config.default import get_config
import open3d as o3d
from zero.expForwardKinematics.models.lotus.utils.robot_box import RobotBox
from numpy.linalg import inv as matinv
from numpy import array as npa
import warnings

from codebase.z_utils.Rotation import *
from codebase.z_utils.rotation_import import *
warnings.filterwarnings("ignore")
'''
joint1: -166, 166
joint2: -101, 101
joint3: -166, 166
joint4: -176, -4
joint5: -166, 166
joint6: -1, 215
joint7: -166,166
'''
# np.set_printoptions(precision=4, suppress=True)


def HT2Pose(T):
    out = np.hstack((T[:3, 3], mat2euler(T[:3, :3])))
    # radian to degree
    out[3:] = np.degrees(out[3:])

    return out


# def Batch_HT2Pose(T):
#     return np.array([HT2Pose(T[i]) for i in range(T.shape[0])])


def printT(name, T):
    print("====================================")
    out = np.array([HT2Pose(T[i]) for i in range(T.shape[0])])
    # out = np.around(out, 2)
    print(f'{name}\n', out)


headless = 1
LINK_NUM = 8

# theta_deg = [-30, 30, 30, -30, -30, 30, 30]
# theta_deg = [-15, 15, 15, -15, -15, 15, 15]
theta_deg = [15, 0, 0, 0, 0, 0, 0]
# theta_deg = [30, 0, 0, 0, 0, 0, 0]
# theta_deg = [45, 0, 0, 0, 0, 0, 0]
# theta_deg = [0, 0, 0, 0, 0, 0, 0]

theta = np.radians(theta_deg)
config_path = "/data/zero/zero/expForwardKinematics/config/DP.yaml"
config = get_config(config_path)
config.defrost()
config.TRAIN_DATASET.rm_robot = False
config.freeze()


def get_Frames_Transformation(theta):
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

    d = [0.333, 0, 0.316, 0, 0.384, 0, 0, ]
    a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, ]
    alpha = [0, -math.pi / 2, math.pi / 2, math.pi / 2,
             -math.pi / 2, math.pi / 2, math.pi / 2, ]

    T_i_1_i_list = []
    for i in range(7):
        T = dh_modified_transform(alpha[i], a[i], theta[i], d[i])
        T_i_1_i_list.append(T)

    # 转置旋转角
    # for i in range(7):
    #     T_i_1_i_list[i][:3, :3] = T_i_1_i_list[i][:3, :3].T
    # # 专职
    joint_locations = []
    T_base = np.array([
        [1, 0, 0, -0.2677189],
        [0, 1, 0, -0.00628856],
        [0, 0, 1, 0.74968816],
        [0, 0, 0, 1]])
    T_cumulative = T_base
    T_i_list = []
    for T in T_i_1_i_list:
        T_cumulative = T_cumulative @ T
        T_i_list.append(copy(T_cumulative))

        pos = T_cumulative[:3, 3]
        joint_locations.append(pos)

    T_i_1_i_list = np.array(T_i_1_i_list)
    T_i_list = np.array(T_i_list)
    joint_locations = np.array(joint_locations)
    return T_i_1_i_list, T_i_list, joint_locations


def simulator():
    obs_processor = ObsProcessorPtv3(config, train_flag=0)
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
        # action = np.random.randn(*action.shape

        action = np.hstack((theta, 1))
        action[3] = action[3] + math.radians(-4)
        obs, reward, terminate = task.step(action)

        # 1. get the position of links, but I dont know which is the part of the robot
        Link_Pose = []
        for link_id in range(1, LINK_NUM):
            Link_Pose.append(obs.misc[f'Panda_link{link_id}_respondable_pose'])
            # break
        Link_Pose = np.array(Link_Pose)
        # print(link_pose_theta_all_0)

        # 1.2 get joint poses
        Joint_Position = [env._robot.arm.joints[i].get_joint_position() for i in range(7)]
        Joint_Pose = [env._robot.arm.joints[i].get_pose() for i in range(7)]

        # 2.from env to get the robot info
        if i >= 50:
            # test(obs, link_pose_theta_all_0, joints_locations, joints_position)
            obs_raw = obs_processor.obs_2_obs_raw(obs)
            # static_process = obs_processor.static_process(obs_raw)
            arm_links_info = (obs_raw['bbox'][0], obs_raw['pose'][0])
            Joint_Position = np.array(Joint_Position)
            Joint_Pose = np.array(Joint_Pose)
            Link_Pose = np.array(Link_Pose)

            script(Joint_Position, Joint_Pose, Link_Pose, arm_links_info, obs_processor, obs_raw)
            breakObsProcessorPtv3

# region EXP


def script(Joint_Position_Sim, Joint_Pose_Sim, Link_Pose_Sim, arm_links_info_Sim, obs_processor, obs_raw):
    print("====================================")
    print('theta\n', theta_deg)

    # theory data   # 原料
    T_i1i_theory, T_oi_theory, Joint_Location_theory = get_Frames_Transformation(theta)

    T_oi_theory = copy(T_oi_theory)
    T_op_sim = npa([RT2HT(quat2mat(Link_Pose_Sim[i][3:]), Link_Pose_Sim[i][:3]) for i in range(7)])
    # T_ip_theory_sim = npa([matinv(T_oi_theory[i]) @ T_op_sim[i] for i in range(7)])
    # T_oi_sim = npa([RT2HomoTrans(quat2mat(Joint_Pose_Sim[i][3:]), Joint_Pose_Sim[i][:3]) for i in range(7)])

    pose_gripper_link = arm_links_info_Sim[1]['Panda_gripper_visual']
    bbox_gripper_link = arm_links_info_Sim[0]['Panda_gripper_visual']

    pose_left_finger_link = arm_links_info_Sim[1]['Panda_leftfinger_visual']
    bbox_left_finger_link = arm_links_info_Sim[0]['Panda_leftfinger_visual']

    pose_right_finger_link = arm_links_info_Sim[1]['Panda_rightfinger_visual']
    bbox_right_finger_link = arm_links_info_Sim[0]['Panda_rightfinger_visual']

    T_ok_gripper_link = npa(RT2HT(quat2mat(pose_gripper_link[3:]), pose_gripper_link[:3]))
    T_ok_left_finger_link = npa(RT2HT(quat2mat(pose_left_finger_link[3:]), pose_left_finger_link[:3]))
    T_ok_right_finger_link = npa(RT2HT(quat2mat(pose_right_finger_link[3:]), pose_right_finger_link[:3]))

    T_ik_gripper_link = matinv(T_oi_theory[-1]) @ T_ok_gripper_link
    T_ik_left_finger_link = matinv(T_oi_theory[-1]) @ T_ok_left_finger_link
    T_ik_right_finger_link = matinv(T_oi_theory[-1]) @ T_ok_right_finger_link

    JPose_gripper_link = HT2Pose(T_ik_gripper_link)
    JPose_left = HT2Pose(T_ik_left_finger_link)
    JPose_right = HT2Pose(T_ik_right_finger_link)
    print('JPose\n', JPose_gripper_link)
    print('JPose\n', JPose_left)
    print('JPose\n', JPose_right)

    # print([degrees(Joint_Position_Sim[i]) for i in range(7)])

    # bbox = []
    # for i in range(7):
    #     this_bbox = npa(arm_links_info_Sim[0][f'Panda_link{i+1}_respondable'])
    #     bbox.append(this_bbox)
    # bbox = npa(bbox)

    # np.save(f'bbox_{theta_deg}.npy', bbox)
    # print(bbox)
    # T_ip_theory_sim = npa([matinv(T_oi_theory[i]) @ T_op_sim[i] for i in range(7)])
    # printT('T_ip_theory_sim', T_ip_theory_sim)
    # T_ip_theory_sim_pose = np.array([HT2Pose(T_ip_theory_sim[i]) for i in range(7)])
    # np.save(f'T_ip_theory_sim_{theta_deg}.npy', T_ip_theory_sim_pose)
    # print('T_ip_theory_sim_pose\n', T_ip_theory_sim_pose)

    # T_ip_mean = np.array([
    #     [-0.0001, -0.0347, -0.0752, -162.8693, 0.0033, 0.2122],
    #     [0.0, -0.0766, 0.0344, -72.9831, 0.2349, -178.696],
    #     [0.0333, 0.0266, -0.0412, -23.1133, 36.1784, -73.6655],
    #     [-0.0495, 0.0425, 0.0267, 67.4364, 35.2228, 105.8928],
    #     [-0.0012, 0.043, -0.109, -14.2746, -0.5962, -90.9492],
    #     [0.0425, 0.0152, 0.01, 2.82, -77.4572, 2.8245],
    #     [0.0136, 0.0117, 0.0787, 89.277, -45.0221, 88.992]
    # ])
    # T_ip_mean = npa([RT2HT(euler2mat(np.radians(T_ip_mean[i, 3:])), T_ip_mean[i, :3])for i in range(7)])

    # T_op_theory_1 = npa([T_oi_theory[i] @ T_ip_theory_sim[i] for i in range(7)])
    # T_op_theory = npa([T_oi_theory[i] @ T_ip_mean[i] for i in range(7)])

    # # printT('T_op_theory', T_op_theory)
    # # printT('T_op_theory_1', T_op_theory_1)
    # # bbox = arm_links_info_Sim[0]
    # bbox = []
    # for i in range(7):
    #     this_bbox = npa(arm_links_info_Sim[0][f'Panda_link{i+1}_respondable'])
    #     bbox.append(this_bbox)

    # # obbox = o3d.geometry.OrientedBoundingBox(
    # #     link_pose[:3], link_rot, link_bbox[1::2] - link_bbox[::2]
    # # )

    # obbox = []
    # test = euler2mat((0, 0, 0))
    # for i in range(7):
    #     rot = (T_op_theory[i, :3, :3]).reshape(3, 3)
    #     test1 = bbox[i][1::2]
    #     test2 = bbox[i][::2]
    #     extend = (bbox[i][1::2] - bbox[i][::2])
    #     center = T_op_theory[i, :3, 3].reshape(3, 1)
    #     print(extend)
    #     this_obbox = o3d.geometry.OrientedBoundingBox(
    #         center=center, R=rot, extent=extend
    #     )
    #     obbox.append(this_obbox)

    # # obbox = o3d.geometry.OrientedBoundingBox(
    # #     , link_rot_2, link_bbox[1::2] - link_bbox[::2]
    # # )

    # ######################################################
    # ######################################################
    # ######################################################
    # # For Visualization
    # static_process = obs_processor.static_process(obs_raw)
    # # pcd
    # xyz = copy(static_process['xyz'][0])
    # rgb = copy(static_process['rgb'][0])
    # # frames location
    # location_frame = T_oi_theory[:, :3, 3]
    # xyz = np.vstack([location_frame, xyz])
    # this_rgb = np.zeros((len(location_frame), 3))
    # this_rgb[:, :2] = 125
    # rgb = np.vstack([this_rgb, rgb])

    # # link location

    # link_location = T_op_theory[:, :3, 3]
    # xyz = np.vstack([link_location, xyz])
    # link_rgb = np.zeros((len(link_location), 3))
    # link_rgb[:, 2] = 255
    # rgb = np.vstack([link_rgb, rgb])

    # #################### SIMULATION######################

    # # link pose
    # link_pose_point = Link_Pose_Sim[:, :3]
    # xyz = np.vstack([link_pose_point, xyz])
    # link_pose_rgb = np.zeros((LINK_NUM - 1, 3))
    # link_pose_rgb[:, 0] = 255
    # rgb = np.vstack([link_pose_rgb, rgb])

    # # joint pose
    # joint_locations = Joint_Pose_Sim[:, :3]
    # jl = np.array(joint_locations)
    # xyz = np.vstack([jl, xyz])
    # joint_pose_rgb = np.zeros((7, 3))
    # joint_pose_rgb[:, 1] = 255
    # rgb = np.vstack([joint_pose_rgb, rgb])
    # arm_links_info = (obs_raw['bbox'][0], obs_raw['pose'][0])

    # robot_box = RobotBox(
    #     arm_links_info, keep_gripper=False,
    #     env_name='rlbench', selfgen=True
    # )
    # obb = robot_box.robot_obboxes
    # for single_obb in obb:
    #     single_obb.color = (0, 0, 1)

    # for single_obb in obbox:
    #     single_obb.color = (1, 0, 0)

    # # visualize   # 先可视化一下吧，把做到什么程度显示出来。
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    # o3d.visualization.draw_geometries([pcd, *obb, *obbox])
    # o3d.visualization.draw_geometries([pcd, *obbox])


simulator()
