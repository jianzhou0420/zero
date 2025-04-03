import copy
import numpy as np
import math
from math import cos, sin
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
from math import cos, sin


JOINT_NUM = 8


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


theta = [0, 0, 0, 0, 0, 0, 0]
d = [0.333, 0, 0.316, 0, 0.384, 0, 0, ]
a = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, ]
alpha = [0, -math.pi / 2, math.pi / 2, math.pi / 2,
         -math.pi / 2, math.pi / 2, math.pi / 2, ]


T_list = []
for i in range(7):
    T = dh_modified_transform(alpha[i], a[i], theta[i], d[i])
    T_list.append(T)


joint_positions = []  # 存放各个关节的坐标
T_cumulative = np.eye(4)  # 累积变换矩阵
for T in T_list:
    T_cumulative = T_cumulative @ T   # 或 np.dot(T_cumulative, T)
    pos = T_cumulative[:3, 3]  # 提取平移部分 (x, y, z)
    joint_positions.append(pos)

'''
joint1: -166, 166
joint2: -101, 101
joint3: -166, 166
joint4: -176, -4
joint5: -166, 166
joint6: -1, 215
joint7: -166,166
'''

# 定义点集，每行代表一个点的 (x, y, z) 坐标
points = np.array(joint_positions)
points = np.vstack([[0, 0, 0], points])
print(points)


# 创建绘图窗口和3D坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

# 如果需要将这些点按顺序连接起来（如连成一条线）
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r', linestyle='-')

# for i, (x, y, z) in enumerate(points):
#     ax.text(x, y, z, f'frame{i}x{x}y{y}z{z}', fontsize=10, color='black')

# 添加坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
plt.show()
