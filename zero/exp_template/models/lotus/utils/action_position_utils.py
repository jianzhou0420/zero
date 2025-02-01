import numpy as np
import einops
import collections
import time

# 中心点(0.25,0,0.7520)
# sizeSize (Bounding Box): [-0.32499998807907104, 0.32499998807907104, -0.45500004291534424, 0.45500004291534424, -0.0, 0.0]
BIN_SPACE = [[-0.325 + 0.25, 0.325 + 0.25], [-0.455, 0.455], [0.7520 + 0, 0.7520 + 0.5]]


def get_disc_gt_pos_prob(
    xyz, gt_pos, bin_strategy, pos_bin_size=0.01, pos_bins=50, heatmap_type='plain', robot_point_idxs=None, bin_space=None
):
    '''
    heatmap_type:
        - plain: the same prob for all voxels with distance to gt_pos within pos_bin_size
        - dist: prob for each voxel is propotional to its distance to gt_pos
    '''

    if bin_strategy == 'nonlinear':
        scaling_factor = 1.8
        bins = np.arange(-pos_bins, pos_bins)
        shift = np.sign(bins) * (np.abs(bins) ** scaling_factor) * pos_bin_size
        cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None]  # (npoints, 3, pos_bins*2)
        dists = np.abs(gt_pos[None, :, None] - cands_pos)
        dists = einops.rearrange(dists, 'n c b -> c (n b)')

        if heatmap_type == 'plain':
            disc_pos_prob = np.zeros((3, xyz.shape[0] * pos_bins * 2), dtype=np.float32)
            disc_pos_prob[dists < 0.01] = 1
            if robot_point_idxs is not None and len(robot_point_idxs) > 0:
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
                disc_pos_prob[:, robot_point_idxs] = 0
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
            for i in range(3):
                if np.sum(disc_pos_prob[i]) == 0:
                    disc_pos_prob[i, np.argmin(dists[i])] = 1
            disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)
            # disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b')
        else:
            disc_pos_prob = 1 / np.maximum(dists, 1e-4)
            # TODO
            # disc_pos_prob[dists > 0.02] = 0
            disc_pos_prob[dists > pos_bin_size] = 0  # 距离gt_pos大于pos_bin_size的点概率为0
            if robot_point_idxs is not None and len(robot_point_idxs) > 0:  # 去掉机器人的点
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
                disc_pos_prob[:, robot_point_idxs] = 0
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
            for i in range(3):
                if np.sum(disc_pos_prob[i]) == 0:
                    disc_pos_prob[i, np.argmin(dists[i])] = 1
            disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)
    elif bin_strategy == 'global':
        # 因为heatmap只有一个target，所以，不需要每个xyz都算，只要算x，y，z三个轴上的概率即可

        x_bins_location = np.arange(bin_space[0][0], bin_space[0][1], pos_bin_size)
        y_bins_location = np.arange(bin_space[1][0], bin_space[1][1], pos_bin_size)
        z_bins_location = np.arange(bin_space[2][0], bin_space[2][1], pos_bin_size)
        # bins natually have coordinates

        x_target, y_target, z_target = gt_pos
        x_dist = np.clip(np.abs(x_bins_location - x_target), 1e-4, 1)  # avoid 0
        y_dist = np.clip(np.abs(y_bins_location - y_target), 1e-4, 1)
        z_dist = np.clip(np.abs(z_bins_location - z_target), 1e-4, 1)

        x_prob = 1 / x_dist
        y_prob = 1 / y_dist
        z_prob = 1 / z_dist
        disc_pos_prob = [x_prob, y_prob, z_prob]  # 因为结构不一样，所以不用np.stack
        return x_prob, y_prob, z_prob

    else:  # default,lotus
        shift = np.arange(-pos_bins, pos_bins) * pos_bin_size  # (pos_bins*2, )
        cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None]  # (npoints, 3, pos_bins*2)
        dists = np.abs(gt_pos[None, :, None] - cands_pos)  # (npoints, 3, pos_bins*2)
        dists = einops.rearrange(dists, 'n c b -> c (n b)')  # (3, npoints*pos_bins*2)

        if heatmap_type == 'plain':
            disc_pos_prob = np.zeros((3, xyz.shape[0] * pos_bins * 2), dtype=np.float32)
            disc_pos_prob[dists < 0.01] = 1
            if robot_point_idxs is not None and len(robot_point_idxs) > 0:
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
                disc_pos_prob[:, robot_point_idxs] = 0
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
            for i in range(3):
                if np.sum(disc_pos_prob[i]) == 0:
                    disc_pos_prob[i, np.argmin(dists[i])] = 1
            disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)
            # disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b')
        else:
            disc_pos_prob = 1 / np.maximum(dists, 1e-4)
            # TODO
            # disc_pos_prob[dists > 0.02] = 0
            disc_pos_prob[dists > pos_bin_size] = 0
            if robot_point_idxs is not None and len(robot_point_idxs) > 0:
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c (n b) -> c n b', n=xyz.shape[0])
                disc_pos_prob[:, robot_point_idxs] = 0
                disc_pos_prob = einops.rearrange(disc_pos_prob, 'c n b -> c (n b)')
            for i in range(3):
                if np.sum(disc_pos_prob[i]) == 0:
                    disc_pos_prob[i, np.argmin(dists[i])] = 1
            disc_pos_prob = disc_pos_prob / np.sum(disc_pos_prob, -1, keepdims=True)

    return disc_pos_prob


def get_best_pos_from_disc_pos(disc_pos_prob, xyz, bin_strategy=None, bin_space=None, pos_bin_size=0.01, pos_bins=50, best='max', topk=1000):
    '''Args:
        disc_pos_prob: (3, npoints*pos_bins*2)
        xyz: (npoints, 3)
    '''
    if bin_strategy == 'global':
        x_bins_location = np.arange(bin_space[0][0], bin_space[0][1], pos_bin_size)
        y_bins_location = np.arange(bin_space[1][0], bin_space[1][1], pos_bin_size)
        z_bins_location = np.arange(bin_space[2][0], bin_space[2][1], pos_bin_size)
        x_prob, y_prob, z_prob = disc_pos_prob
        x_pos = np.argmax(x_prob)
        y_pos = np.argmax(y_prob)
        z_pos = np.argmax(z_prob)

        return best_pos
    if best == 'max_scale_bins':

        pos_bins = 15
        pos_bin_size = 0.001
        scaling_factor = 1.8

        bins = np.arange(-pos_bins, pos_bins)
        shift = np.sign(bins) * (np.abs(bins) ** scaling_factor) * pos_bin_size

        cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None]  # (npoints, 3, pos_bins*2)
        cands_pos = einops.rearrange(cands_pos, 'n c b -> c (n b)')  # (3, npoints*pos_bins*2)
        idxs = np.argmax(disc_pos_prob, -1)
        best_pos = cands_pos[np.arange(3), idxs]

    else:  # author's code
        assert best in ['max', 'ens1']
        shift = np.arange(-pos_bins, pos_bins) * pos_bin_size  # (pos_bins*2, )
        cands_pos = np.stack([shift] * 3, 0)[None, :, :] + xyz[:, :, None]  # (npoints, 3, pos_bins*2)

        if best == 'max':
            cands_pos = einops.rearrange(cands_pos, 'n c b -> c (n b)')  # (3, npoints*pos_bins*2)
            idxs = np.argmax(disc_pos_prob, -1)
            best_pos = cands_pos[np.arange(3), idxs]

        elif best == 'ens1':
            # st = time.time()
            cands_pos = einops.rearrange(cands_pos, 'n c b -> c (n b)')  # (3, npoints*pos_bins*2)
            # disc_pos_prob = torch.from_numpy(disc_pos_prob)
            # disc_pos_prob = torch.softmax(disc_pos_prob, -1).numpy()
            cands_pos_voxel = np.round(cands_pos / 0.005).astype(np.int32)  # (3, npoints*pos_bins*2)
            idxs = np.argsort(-disc_pos_prob, -1)  # [:, :topk]
            best_pos = []
            for i in range(3):
                sparse_values = collections.defaultdict(int)
                # for k in idxs[i, :topk]:
                for k in idxs[i]:
                    sparse_values[cands_pos_voxel[i, k].item()] += disc_pos_prob[i, k]
                best_pos_i, best_value = None, -np.inf
                for k, v in sparse_values.items():
                    if v > best_value:
                        best_value = v
                        best_pos_i = k
                best_pos.append(best_pos_i * 0.005)
                # print(i, 'npoints', xyz.shape, 'uniq voxel', len(sparse_values), best_value)
            best_pos = np.array(best_pos)

    return best_pos
