from zero.expForwardKinematics.ReconLoss.ForwardKinematics import FrankaEmikaPanda
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


class visualizor:

    @staticmethod
    def visualize_eePose(eePose, return_o3d=False):
        franka = FrankaEmikaPanda()
        pcd = o3d.geometry.PointCloud()

        # Extract position and quaternion
        position = np.array(eePose[:3])  # (x, y, z)
        quaternion = np.array(eePose[3:])  # (qx, qy, qz, qw)

        # Create rotation from quaternion
        r = R.from_quat(quaternion)

        # Create a coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Apply the rotation to the coordinate frame
        coordinate_frame.rotate(r.as_matrix(), center=(0, 0, 0))

        # Translate the coordinate frame to the position
        coordinate_frame.translate(position)
        if return_o3d:
            return coordinate_frame
        else:
            # Visualize the pose with Open3D
            o3d.visualization.draw_geometries([coordinate_frame])

    @staticmethod
    def visualize_JP(JP, return_o3d=False):
        franka = FrankaEmikaPanda()
        bbox_link, bbox_other = franka.theta2obbox(JP)
        bbox_all = bbox_link + bbox_other
        if return_o3d:
            return bbox_all
        else:
            o3d.visualization.draw_geometries([*bbox_all])

    @staticmethod
    def visualize_eePose_JP(eePose, JP, return_o3d=False):
        bbox = visualizor.visualize_JP(JP, return_o3d=True)
        eePose_1 = visualizor.visualize_eePose(eePose, return_o3d=True)
        if return_o3d:
            return bbox, eePose_1
        else:
            o3d.visualization.draw_geometries([*bbox, eePose_1])


if __name__ == '__main__':
    import pickle
    with open('/data/zero/1_Data/B_Preprocess/eePoseJP/data.pkl', 'rb')as f:
        data = pickle.load(f)
    single_data = data[0]
    JP = single_data[1][:-1]
    print(JP)
    eePose = single_data[0][:-1]
    print(eePose)
    visualizor.visualize_eePose_JP(eePose, JP)
