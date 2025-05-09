import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# Example pose: (x, y, z, qx, qy, qz, qw)
pose = [1.0, 2.0, 3.0, 0.7071, 0.0, 0.7071, 0.0]  # example values

# Extract position and quaternion
position = np.array(pose[:3])  # (x, y, z)
quaternion = np.array(pose[3:])  # (qx, qy, qz, qw)

# Create rotation from quaternion
r = R.from_quat(quaternion)

# Create a coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Apply the rotation to the coordinate frame
coordinate_frame.rotate(r.as_matrix(), center=(0, 0, 0))

# Translate the coordinate frame to the position
coordinate_frame.translate(position)


# Visualize the pose with Open3D
o3d.visualization.draw_geometries([coordinate_frame])
