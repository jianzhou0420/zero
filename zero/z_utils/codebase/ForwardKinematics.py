from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
from math import cos, sin


class DHParam:
    theta = []
    d = []
    a = []
    alpha = []


JOINT_NUM = 8


class FrankaPandaForwardKinematics:
    def __init__(self):
        self.DHParam = DHParam()
        self.DHParam.theta = [0, 0, 0.698, 0, 0, 0, 0, 0]
        self.DHParam.d = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
        self.DHParam.a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
        self.DHParam.alpha = [0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2, 0]

    def get_homo_matrix_by_idx(self, idx):
        theta = self.DHParam.theta[idx]
        a = self.DHParam.a[idx]
        d = self.DHParam.d[idx]
        alpha = self.DHParam.alpha[idx]

        r11 = cos(theta)
        r12 = -sin(theta) * cos(alpha)
        r13 = sin(theta) * sin(alpha)
        tx = a * cos(theta)

        r21 = sin(theta)
        r22 = cos(theta) * cos(alpha)
        r23 = -cos(theta) * sin(alpha)
        ty = a * sin(theta)

        r31 = 0
        r32 = sin(alpha)
        r33 = cos(alpha)
        tz = d

        T = np.array([
            [r11, r12, r13, tx],
            [r21, r22, r23, ty],
            [r31, r32, r33, tz],
            [0, 0, 0, 1]
        ])
        return T


fpdk = FrankaPandaForwardKinematics()
homo_matrix = []

for i in range(8):
    homo_matrix.append(fpdk.get_homo_matrix_by_idx(i))
homo_matrix = np.array(homo_matrix)

# 创建8个独立的单位矩阵
homo_matrix_0_to_i = [np.identity(4) for _ in range(8)]

for i in range(len(homo_matrix)):
    for j in range(i + 1):
        homo_matrix_0_to_i[i] = homo_matrix_0_to_i[i] @ homo_matrix[j]

homo_matrix_0_to_i = np.array(homo_matrix_0_to_i)
print("各关节相对于基坐标系的齐次变换矩阵：")
print(homo_matrix_0_to_i)

translations = homo_matrix_0_to_i[:, :3, 3]
print("平移部分：")
print(translations)


# 定义点集，每行代表一个点的 (x, y, z) 坐标
points = translations

# 创建绘图窗口和3D坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

# 如果需要将这些点按顺序连接起来（如连成一条线）
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r', linestyle='-')

for i, (x, y, z) in enumerate(points):
    ax.text(x, y, z, f'joint{i}', fontsize=10, color='black')

# 添加坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


link_pose_theta_all_0 = [
    [-0.301, -0.013],  # link0
]
