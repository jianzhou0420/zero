import pickle
from scipy.spatial.transform import Rotation as R
import torch
print(torch.cuda.is_available())  # Should print True if CUDA is available


# def query_quat_order(q):
#     euler_xyzw = R.from_quat(q, scalar_first=False).as_euler('xyz', degrees=True)
#     euler_wxyz = R.from_quat(q, scalar_first=True).as_euler('xyz', degrees=True)
#     print('if xyzw:', euler_xyzw)
#     print('if wxyz:', euler_wxyz)


# # data_path = "/media/jian/ssd4t/zero/1_Data/B_Preprocess/DP/keypose/singleVar/train/put_groceries_in_cupboard/variation1/episode46/data.pkl"
# # with open(data_path, 'rb') as f:
# #     data = pickle.load(f)
# # print(data.keys())
# # eePose = data['eePose_hist'][1]
# # quat = eePose[:, 3:7]
# # query_quat_order(quat)


# euler = [15, 45, 60]
# quat = R.from_euler('xyz', euler, degrees=True).as_quat(scalar_first=True)
# matrix = R.from_euler('xyz', euler, degrees=True).as_matrix()


# def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
#     """
#     Convert rotations given as quaternions to rotation matrices.

#     Args:
#         quaternions: quaternions with real part first,
#             as tensor of shape (..., 4).

#     Returns:
#         Rotation matrices as tensor of shape (..., 3, 3).
#     """
#     r, i, j, k = torch.unbind(quaternions, -1)
#     # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
#     two_s = 2.0 / (quaternions * quaternions).sum(-1)

#     o = torch.stack(
#         (
#             1 - two_s * (j * j + k * k),
#             two_s * (i * j - k * r),
#             two_s * (i * k + j * r),
#             two_s * (i * j + k * r),
#             1 - two_s * (i * i + k * k),
#             two_s * (j * k - i * r),
#             two_s * (i * k - j * r),
#             two_s * (j * k + i * r),
#             1 - two_s * (i * i + j * j),
#         ),
#         -1,
#     )
#     return o.reshape(quaternions.shape[:-1] + (3, 3))


# matrix2 = quaternion_to_matrix(torch.tensor(quat).unsqueeze(0))
# print('matrix:', matrix)
# print('matrix2:', matrix2)


with open("/media/jian/ssd4t/zero/1_Data/A_Selfgen/keypose/singleVar/variation_descriptions.pkl", 'rb') as f:
    data = pickle.load(f)
print(data)
