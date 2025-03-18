
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pickle
from sklearn.cluster import KMeans as skKMeans
from sklearn.cluster import DBSCAN as skDBSCAN


class ClusterManager:
    def __init__(self):
        self.clusters = []

    def DBSCAN_o3d(self, pc):
        # 1. 加载点云数据,numpy
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        print("点云包含 %d 个点" % np.asarray(pcd.points).shape[0])

        # 2. 可选预处理：下采样（体素滤波）
        # voxel_size = 0.001  # 根据数据尺度选择合适的体素大小
        # downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downpcd = pcd
        # 3. 使用 DBSCAN 算法进行聚类
        # 参数解释：
        #   eps: 邻域半径
        #   min_points: 构成核心点所需要的最小邻居数
        labels = np.array(
            downpcd.cluster_dbscan(eps=0.03, min_points=100, print_progress=True)
        )
        max_label = labels.max()
        print("聚类数目：", max_label + 1)

        # 4. 为不同的聚类赋予不同的颜色（噪声点默认标记为 -1）
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # 噪声点设置为黑色
        downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # 5. 可视化聚类结果
        o3d.visualization.draw_geometries([downpcd])

        print('test1')


sort_shape_path = "/data/zero/1_Data/B_Preprocess/train/0.005_sort_shape/place_shape_in_shape_sorter/variation0/episodes/episode0/data.pkl"
insert_peg_path = '/data/zero/1_Data/B_Preprocess/train/insert_0.005/insert_onto_square_peg/variation0/episodes/episode0/data.pkl'
with open(sort_shape_path, 'rb') as f:
    data = pickle.load(f)

cm = ClusterManager()

for i in range(len(data['xyz'])):
    cm.DBSCAN_o3d(data['xyz'][i])
