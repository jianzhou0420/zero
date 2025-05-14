import open3d as o3d
import numpy as np
import torch


def make_pcd(pcd, rgb=None) -> o3d.geometry.PointCloud:
    if isinstance(pcd, list) or isinstance(pcd, np.ndarray):
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(pcd)
        if rgb is not None:
            o3dpcd.colors = o3d.utility.Vector3dVector(rgb)
        pcd = o3dpcd
    elif isinstance(pcd, o3d.geometry.PointCloud):
        pcd = pcd
    else:
        raise TypeError(f'pcd should be list or open3d.geometry.PointCloud, but got {type(pcd)}')
    return pcd


def extract_pcd(pcd):
    xyz = np.asarray(pcd.points)
    if len(pcd.colors) > 0:
        rgb = np.asarray(pcd.colors)
    return xyz, rgb


def pcd_remove_outliers(pcd, nb_neighbors=10, std_ratio=2.0, return_pcd=False):
    pcd = make_pcd(pcd)
    mask = np.zeros((len(pcd.points),), dtype=bool)

    outpcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    mask[ind] = True
    if return_pcd == True:
        return outpcd, mask
    xyz = np.asarray(outpcd.points)
    return xyz, mask


def pcd_visualize(xyz, rgb):
    if torch.is_tensor(xyz):
        xyz = xyz.cpu().numpy()
    if torch.is_tensor(rgb):
        rgb = rgb.cpu().numpy()
    if np.max(rgb) > 1:
        rgb = rgb / 255
    pcd = make_pcd(xyz, rgb)
    o3d.visualization.draw_geometries([pcd], width=800, height=600, window_name="Point Cloud Visualization")


def pcd_voxel_downsample(xyz, rgb=None, voxel_size=0.01):
    pcd = make_pcd(pcd, rgb)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz, rgb = extract_pcd(pcd)
    return xyz, rgb


def pcd_uniform_downsample(xyz, rgb=None, every_k_points=5):
    pcd = make_pcd(xyz, rgb)
    pcd = pcd.uniform_down_sample(every_k_points=every_k_points)
    xyz, rgb = extract_pcd(pcd)
    return xyz, rgb


def pcd_random_downsample_by_ratio(xyz, rgb=None, ratio=0.5):
    pcd = make_pcd(xyz, rgb)
    pcd = pcd.random_down_sample(sampling_ratio=ratio)
    xyz, rgb = extract_pcd(pcd)
    return xyz, rgb


def pcd_fps_downsample(xyz, rgb=None, num_points=1000):
    pcd = make_pcd(xyz, rgb)
    pcd = pcd.farthest_point_sampling(num_points=num_points)
    xyz, rgb = extract_pcd(pcd)
    return xyz, rgb


def pcd_random_downsample_by_num(xyz, rgb=None, num_points=1000, return_idx=False):
    # draw 'num_points' unique indices from [0 .. N-1]
    idxs = np.random.choice(xyz.shape[0], num_points, replace=False)

    if return_idx:
        return idxs
    # apply them to xyz
    sampled_xyz = xyz[idxs]

    # if you have colors, apply the *same* idxs to rgb
    sampled_rgb = rgb[idxs] if rgb is not None else None

    return sampled_xyz, sampled_rgb
