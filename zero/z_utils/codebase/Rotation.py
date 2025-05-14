'''
Rotation:
    - euler2mat, euler2quat
    - mat2euler, quat2mat
    - quat2mat, mat2quat
注意！ wxyz和xyzw的区别！ transforms3d用的是wxyz！！！！我这里用的scipy，是xyzw！！！！
三种表达方式的相互转换


PosEuler: position + euler angles

eePose: 
'''
from numpy import radians, degrees
import numpy as np
from scipy.spatial.transform import Rotation as R


def quat2euler(q):

    return R.from_quat(q).as_euler('XYZ')


def quat2mat(q):
    return R.from_quat(q).as_matrix()


def mat2euler(mat):
    return R.from_matrix(mat).as_euler('XYZ')


def mat2quat(mat):
    return R.from_matrix(mat).as_quat()


def euler2quat(euler):
    return R.from_euler('XYZ', euler).as_quat()


def euler2mat(euler):
    return R.from_euler('XYZ', euler).as_matrix()


def RT2HT(R, T):
    HT = np.eye(4)
    HT[:3, :3] = R
    HT[:3, 3] = T
    return HT


def HT2PosEuler(T):
    out = np.hstack((T[:3, 3], mat2euler(T[:3, :3])))
    # radian to degree
    out[3:] = np.degrees(out[3:])

    return out


def PosEuler2HT(PosEuler):
    HT = RT2HT(euler2mat(radians(PosEuler[3:])), PosEuler[:3])
    return HT


def eePose2HT(eePose):
    HT = RT2HT(quat2mat(eePose[3:]), eePose[:3])
    return HT


def HT2eePose(T):
    out = np.hstack((T[:3, 3], mat2quat(T[:3, :3])))
    return out


if __name__ == '__main__':
    import torch
    # test quat2euler
    q = torch.tensor([0.99810947, 0.06146124, 0.0, 0.0])
    print(quat2euler(q))
