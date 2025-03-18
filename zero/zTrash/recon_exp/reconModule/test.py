
'''
this is a prototype for the point reconstruction module for robot manipulation,不考虑efficiency和memory useage

系统接收什么？
4张RGBD照片，每张512*512个点，每个点有xyz和rgb信息

intialize:先从4张照片，对每个物体进行重建，每个物体有自己的local coordinate system

Class PerceptionManager: 一个global coordinate system负责管理所有物体和时间的位置信息
Class PointObject: 一个local coordinate system

物体的重建和时间是无关的。
时间只与物体的位置有关。


工作的困难点在于什么，首先我们没有GT的

'''
import pickle
import numpy as np
from zero.expBins.models.lotus.utils.robot_box import RobotBox


class PointObject:
    def __init__(self, bbox, voxel_size, feature_dim):
        '''
        Args:
            bbox (list): Bounding box coordinates (x_min, y_min, z_min, x_max, y_max, z_max).
            voxel_size (float): Voxel size.
        '''
        self.x_voxels = (bbox[3] - bbox[0]) / voxel_size
        self.y_voxels = (bbox[4] - bbox[1]) / voxel_size
        self.z_voxels = (bbox[5] - bbox[2]) / voxel_size

        self.voxels = np.zeros((self.x_voxels, self.y_voxels, self.z_voxels, feature_dim))

    ################################
    ####### public methods##########
    ################################

    def new_points(self, frame):
        pass

    def get_voxels(self):
        return self.voxels
    ################################
    ####### private methods##########
    ################################


class PerceptionManager:
    def __init__(self, workspace, voxel_size, feature_dim):
        '''
        因为点云的xyz是动态的，所以需要
        '''

        self.WORKSPACE = workspace

    def initialize(self, RGBDs: np.ndarray, arm_links_info):
        '''
        RGBDs: 4张RGBD照片, 每张512*512个点，每个点有xyz和rgb信息
        '''
        pc = self._preprocess(RGBDs, arm_links_info)

        point_objects = self.main_process(pc)

        pass

    def update(self, RGBDs: np.ndarray, time: int):
        '''
        RGBDs: 4张RGBD照片, 每张512*512个点，每个点有xyz和rgb信息


        logic：两个PointObject的对齐，然后更新
        '''

        pass

    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
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

    def _preprocess(self, RGBDs: np.ndarray, arm_links_info, rm_robot):
        '''
        Args:
            RGBD: 512*512个点，每个点有xyz和rgb信息

            先lotus preprocess, 然后cluster，


        return: lotus一样的点云处理 but without voxelization
        '''
        xyz = RGBDs[:, :, 0:3].reshape(-1, 3)
        rgb = RGBDs[:, :, 3:6].reshape(-1, 3)

        # 1. remove outside worksapce

        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
            (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
            (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])

        # 2. remove table
        in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])

        xyz = xyz[in_mask]
        rgb = rgb[in_mask]

        # 3. remove robot

        mask = self._get_mask_with_robot_box(xyz, arm_links_info, 'box_keep_gripper')
        xyz = xyz[mask]
        rgb = rgb[mask]
        pc = np.concatenate((xyz, rgb), axis=1)
        return pc

    def main_process(self, points: np.ndarray):
        '''
        Args:
            RGBD: 512*512个点，每个点有xyz和rgb信息

            先lotus preprocess, 然后cluster，


        return: list of PointObject with their global position
        '''

        pass


if __name__ == '__main__':
    example_episodes_path = '/data/zero/1_Data/B_Preprocess/train/insert_0.005/insert_onto_square_peg/variation0/episodes/episode0/data.pkl'
    with open(example_episodes_path, 'rb') as f:
        data = pickle.load(f)
    print(data.keys())
