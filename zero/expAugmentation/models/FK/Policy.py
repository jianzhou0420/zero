'''
ForwardKinematics Policy
'''
from torch.nn.utils.rnn import pad_sequence
import math
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
import torch.nn as nn
from zero.expAugmentation.models.dp2d.components.PointTransformerV3.model import (
    PointTransformerV3, offset2bincount, offset2batch
)
from zero.expAugmentation.models.lotus.PointTransformerV3.model_ca import PointTransformerV3CA
from zero.expAugmentation.config.default import build_args
from zero.expAugmentation.models.Base.BaseAll import BaseActionHead, BaseFeatureExtractor, BasePolicy
import numpy as np
import einops
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from zero.expAugmentation.models.Base.BaseAll import BasePolicy
from zero.z_utils.joint_position import normaliza_JP, denormalize_JP
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from zero.expAugmentation.models.Base.BaseAll import BaseActionHead

from zero.expAugmentation.config.default import get_config
import dgl.geometry as dgl_geo
from codebase.z_model.attentionlayer import SelfAttnFFW, CrossAttnFFW, PositionalEncoding
from zero.expAugmentation.ReconLoss.ForwardKinematics import FrankaEmikaPanda
# 先不要参数化
# 先不要大改，按照DP的写法来
from torch import Tensor

'''
Policy
ActionHead, Feature Extractor

'''
# ---------------------------------------------------------------
# region 1. Feature Extractor


class FeatureExtractorPTv3CA(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        self.ptv3_model = PointTransformerV3CA(**config.FeatureaExtractor.ptv3)

        self.config = config
        self.txt_fc = nn.Linear(config.Tmp.txt_ft_size, config.Tmp.context_channels)
        self.txt_attn_fc = nn.Linear(config.Tmp.txt_ft_size, 1)

    def forward(self, ptv3_batch):
        '''
        ptv3_batch: dict,dont care

        return [batch, feat_dim] .
        '''
        point = self.ptv3_model(ptv3_batch)

        splited_feature = torch.split(point['feat'], offset2chunk(point['offset']), dim=0)  # alarm
        new_feature = []
        for i, feat in enumerate(splited_feature):
            # max pooling along the
            if self.config.FeatureaExtractor.pool == 'max':
                new_feature.append(torch.max(feat, 0)[0])
            elif self.config.FeatureaExtractor.pool == 'mean':
                new_feature.append(torch.mean(feat, 0))
        new_feature = torch.stack(new_feature, 0)
        return new_feature

    def prepare_ptv3_batch(self, batch):
        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config.Dataset.voxel_size,
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }
        device = batch['pc_fts'].device

        # encode context for each point cloud
        txt_embeds = self.txt_fc(batch['txt_embeds'])
        ctx_embeds = torch.split(txt_embeds, batch['txt_lens'])
        ctx_lens = torch.LongTensor(batch['txt_lens'])

        outs['context'] = torch.cat(ctx_embeds, 0)
        outs['context_offset'] = torch.cumsum(ctx_lens, dim=0).to(device)

        return outs


class FeatureExtractorPTv3Clean(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        self.config = config

        n_heads = config['FK']['ActionHead']['n_heads']
        d_ffw = config['FK']['ActionHead']['d_ffw']
        d_features = config['FK']['FeatureExtractor']['ptv3']['enc_channels'][-1]
        n_features = config['FK']['ActionHead']['n_features']

        self.ptv3_model = PointTransformerV3(**config['FK']['FeatureExtractor']['ptv3'])
        self.cross_attn = CrossAttnFFW(d_features, n_heads, d_ffw, 0.1)

        self.n_features = n_features
        self.d_features = d_features

    def forward(self, ptv3_batch):
        '''
        ptv3_batch: dict,dont care

        return [batch, feat_dim] .
        '''
        point = self.ptv3_model(ptv3_batch)

        pcd_feat = torch.split(point['feat'], offset2chunk(point['offset']), dim=0)  # alarm

        pcd_empty = torch.ones((len(pcd_feat), self.n_features, self.d_features), device=pcd_feat[0].device) * 0.5
        pcd_padded, mask = pad_pcd_features(pcd_feat, self.n_features)
        pcd_feat = self.cross_attn(pcd_empty, pcd_padded, mask=mask)
        return pcd_feat

    def prepare_ptv3_batch(self, batch):

        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config['Dataset']['voxel_size'],
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }

        return outs

# endregion
# ---------------------------------------------------------------
# region 2. ActionHead


class ActionHead(BaseActionHead):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # action head
        d_model = config['FK']['ActionHead']['d_model']
        d_instr = config['FK']['ActionHead']['d_instr']
        d_ffw = config['FK']['ActionHead']['d_ffw']
        n_heads = config['FK']['ActionHead']['n_heads']
        d_pcd_features = config['FK']['FeatureExtractor']['ptv3']['enc_channels'][-1]

        # ActionEmbedding
        self.embed_actionHis = nn.Linear(8, d_model, bias=False)  # TODO:可能可以离散化，用类似普朗克常量的东西
        self.embed_instr = nn.Linear(d_instr, d_model, bias=False)
        self.embed_t = nn.Linear(1, d_model, bias=False)
        self.embed_features = nn.Linear(d_pcd_features, d_model, bias=False)  # TODO:可能可以离散化，用类似普朗克常量的东西
        # 先不对ptv3的feature进行处理

        # position embedding

        self.PE = PositionalEncoding(d_model)

        # prediction layer
        self.SelfAttnList = nn.ModuleList([
            SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
            SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
            SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
            SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
        ])

        self.CrossAttnList = nn.ModuleList([
            CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
            CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
            CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
            CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
        ])

        # prediction layer
        self.JP_futr_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 8)
        )

    def forward(self, pcd_feat, JP_hist, instr, t, JP_futr_noisy):

        t = t.clone()
        t = t.to(torch.float32)
        # for encoder
        embeded_pcd = self.embed_features(pcd_feat)
        embeded_actionHis = self.embed_actionHis(JP_hist)
        embeded_instr = self.embed_instr(instr)
        embeded_t = self.embed_t(t.unsqueeze(-1)).unsqueeze(1)

        # for decoder
        embeded_actionFuture = self.embed_actionHis(JP_futr_noisy)

        feature_all = torch.cat([embeded_pcd, embeded_actionHis, embeded_instr, embeded_t], dim=1)
        feature_all = self.PE(feature_all)

        y = feature_all
        for module in self.SelfAttnList:
            y = module(y)

        x = embeded_actionFuture
        for module in self.CrossAttnList:
            x = module(x, y)

        pred = self.JP_futr_decoder(x)
        return pred


# endregion
# ---------------------------------------------------------------
# region 3. Policy


class Policy(BasePolicy):
    def __init__(self, config):
        super().__init__()

        # Policy itself
        self.ActionHead = ActionHead(config)
        self.FeatureExtractor = FeatureExtractorPTv3Clean(config)
        self.config = config
        self.franka = FrankaEmikaPanda()
        self.action_theta_offset = [0, 0, 0, math.radians(-4), 0, 0, 0]

        # DDPM
        # adapted from
        # https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py

        beta_1 = config['FK']['Policy']['DDPM']['beta_1']
        beta_T = config['FK']['Policy']['DDPM']['beta_T']
        T = config['FK']['Policy']['DDPM']['T']

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))

        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.T = T

    def forward(self, batch):

        # get data
        JP_futr_0 = batch['JP_futr']  # [batch, horizon, action_dim]
        JP_hist = batch['JP_hist']  # [batch, horizon, action_dim]
        instr = batch['instr']
        noncollision_mask = batch['noncollision_mask']  # [batch, horizon, action_dim]
        # feature extraction
        ptv3_batch = self.FeatureExtractor.prepare_ptv3_batch(batch)
        features = self.FeatureExtractor(ptv3_batch)

        # DDPM scheduler
        t = torch.randint(self.T, size=(JP_futr_0.shape[0], ), device=JP_futr_0.device)
        noise = torch.rand_like(JP_futr_0)
        JP_futr_t = self.q_sample(JP_futr_0, t, noise)  # [batch, horizon, action_dim]

        # DDOM predict
        noise_pred = self.ActionHead(features, JP_hist, instr, t, JP_futr_t)
        JP_futr_0_pred = self.inverse_q_sample(JP_futr_t, t, noise_pred)  # [batch, horizon, action_dim]

        # loss
        ddpm_loss = F.mse_loss(JP_futr_0_pred, JP_futr_0, reduction='mean')

        collision_loss = self.collision_loss(batch['pc_fts'][:, :3], batch['npoints_in_batch'], JP_futr_0_pred, noncollision_mask)  # [batch, horizon, action_dim]

        # TODO:collision loss,diffusion loss

        loss = collision_loss + ddpm_loss
        return loss
    ##############################################
    # DDPM
    ##############################################

    def q_sample(self, x_0, t, noise):
        '''
        forward diffusion process, 加噪声
        '''
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t

    def inverse_q_sample(self, x_t, t, noise):
        '''
        inverse diffusion process, 去噪声
        '''
        sqrt_alphas_bar = extract(self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        x_0 = (x_t - sqrt_one_minus_alphas_bar * noise) / sqrt_alphas_bar
        return x_0

    ##############################################
    # Collision
    ##############################################

    def collision_loss(self, xyz, n_xyz, JP, noncollision_mask):
        '''
        Assume the action has shape [B, Horizon,8]
        xyz is a list of Batch, 每个里面有不同的点云.先padding它成为等长的矩阵,记录mask,最后把mask的loss去掉
        '''
        max_len = max(n_xyz)
        P_op, mask = pad_pcd(xyz, n_xyz, max_len)  # [B, max_len, 3]
        noncollision_mask = pad_noncollision_mask(noncollision_mask, max_len)
        # print('torch.sum(mask))', torch.sum(mask))
        # print('torch.sum(noncollision_mask))', torch.sum(noncollision_mask))
        mask = mask & noncollision_mask

        B, H, _ = JP.size()
        JP = einops.rearrange(JP, 'B H d -> (B H) d')  # [B, H, 1, 8]
        T_ok = torch.tensor(np.array([self.franka.get_T_ok(JP[i, :].squeeze())[0] for i in range(JP.size(0))]), device=JP.device, dtype=torch.float32)  # [BH,7,1,4,4]
        T_ok = einops.rearrange(T_ok, '(B H) k d1 d2  -> B H k d1 d2 ', B=B, H=H)  # [B, H, 7, 1, 4, 4]
        bbox = torch.tensor(self.franka.bbox_link_half, device=JP.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-2)  # [1,7,1,3] for boardcasting

        ones_homo = torch.ones((P_op.size(0), P_op.size(1), 1), device=xyz.device)
        P_op_homo = torch.cat((P_op, ones_homo), dim=2)

        # matmul
        P_op_homo = P_op_homo.unsqueeze(-1).unsqueeze(1).unsqueeze(1)  # s means single # (B，1,1,max_len, 4,1)
        T_ok = T_ok.unsqueeze(-3)  # (B，H,Link,1,4,4)
        P_kp = torch.matmul(T_ok, P_op_homo).squeeze()  # (H,Link,xyz,4)
        P_kb = torch.abs(P_kp[..., :3]) - bbox
        invasion = self._get_invasion(P_kb, mask) / B  # TODO: need a more reasonable invasion

        collision_loss = invasion
        return collision_loss

    def _get_invasion(self, P_kb, mask: Tensor):
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # for boardcasting
        idx = P_kb < 0
        idx = idx & mask
        n_negative = len(idx)
        invasion = torch.sum(P_kb[idx]) / n_negative
        # TODO: define a more reasonable invasion
        return invasion

# ---------------------------------------------------------------
# region 0. Some tools


def pad_noncollision_mask(mask, max_len):
    for i, s_mask in enumerate(mask):
        # print(torch.sum(s_mask))
        n_pcd = s_mask.shape[0]
        n_pad = max_len - n_pcd
        pad_tensor = torch.ones([n_pad], device=mask[i].device, dtype=torch.bool)
        mask[i] = torch.cat((mask[i], pad_tensor))

    mask = torch.stack(mask, 0)
    mask = torch.tensor(mask, dtype=torch.bool)
    mask = torch.logical_not(mask)

    return mask


def pad_pcd(xyz: Tensor, n_xyz, max_len):
    '''

    '''
    mask = []
    splited_pcd = list(torch.split(xyz, n_xyz, dim=0))
    for i, s_pcd in enumerate(splited_pcd):
        n_pcd = s_pcd.shape[0]
        n_pad = max_len - n_pcd
        pad_tensor = torch.zeros([n_pad, s_pcd.shape[1]], device=splited_pcd[i].device)
        splited_pcd[i] = torch.cat((splited_pcd[i], pad_tensor))
        s_mask = torch.zeros(max_len, dtype=torch.bool, device=xyz.device)  # 0
        s_mask[:n_pcd] = True
        mask.append(s_mask)
    xyz = torch.stack(splited_pcd, 0)
    mask = torch.stack(mask, 0)  # [B, max_len]
    return xyz, mask


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def pad_pcd_features(pcd_features, max_len):
    # pad the point cloud features to max_len
    batch_size = len(pcd_features)
    padded_pcd_features = torch.zeros(batch_size, max_len, pcd_features[0].size(1), device=pcd_features[0].device)
    mask = torch.ones([batch_size, max_len], dtype=torch.bool, device=pcd_features[0].device)
    for i in range(batch_size):
        cur_len = pcd_features[i].size(0)
        padded_pcd_features[i, :cur_len] = pcd_features[i]
        mask[i, :cur_len] = False

    return padded_pcd_features, mask


def offset2chunk(offset):
    chunk = []
    chunk.append(offset[0])
    for i in range(len(offset) - 1):
        chunk.append(offset[i + 1] - offset[i])
    assert sum(chunk) == offset[-1]
    return chunk


def ptv3_collate_fn(data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])

    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)  # (#all points, 6)

    for key in ['ee_poses', 'gt_actions', 'theta_positions']:
        batch[key] = torch.stack(batch[key], 0)

    # if 'disc_pos_probs' in batch:
    #     batch['disc_pos_probs'] = batch['disc_pos_probs'] # [(3, #all pointspos_bins*2)]

    batch['step_ids'] = torch.LongTensor(batch['step_ids'])

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    return batch
# endregion


def test():
    import pickle
    from math import radians
    from numpy import array as npa
    from einops import rearrange
    from zero.expAugmentation.ObsProcessor.ObsProcessorPtv3 import ObsProcessorPtv3
    import open3d as o3d

    def get_robot_pcd_idx(xyz, obbox):
        points = o3d.utility.Vector3dVector(xyz)
        robot_point_idx = set()
        for box in obbox:
            tmp = box.get_point_indices_within_bounding_box(points)
            robot_point_idx = robot_point_idx.union(set(tmp))
        robot_point_idx = np.array(list(robot_point_idx))
        mask = np.zeros(len(xyz), dtype=bool)
        mask[robot_point_idx] = True
        return mask

    config = get_config('/media/jian/ssd4t/zero/zero/expAugmentation/config/FK.yaml')
    obs_processor = ObsProcessorPtv3(config)
    policy = Policy(config)
    B = 1
    H = 8
    episode_path = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/DA3D/close_jar/variation0/episodes/episode0/data.pkl'

    with open(episode_path, 'rb') as f:
        episode = pickle.load(f)

    print(episode.keys())
    theta_offset = npa([0, 0, 0, radians(-4), 0, 0, 0])

    theta_sim = episode['joint_position_history'][0][-1][:-1]  # 第一帧的关节角度
    theta_the = theta_sim - theta_offset

    rgb = episode['rgb'][0]
    xyz = episode['pcd'][0]

    xyz = rearrange(xyz, 'ncam h w c -> (ncam h w) c')
    rgb = rearrange(rgb, 'ncam h w c -> (ncam h w) c')

    xyz, rgb = obs_processor.within_workspace(xyz, rgb)
    xyz, rgb = obs_processor.remove_table(xyz, rgb)
    xyz, rgb = obs_processor.voxelize(xyz, rgb)

    theta_the = np.hstack([theta_the, 0])

    bbox_link, bbox_other = policy.franka.theta2obbox(theta_the)

    bbox_all = bbox_link + bbox_other

    pcd_idx = get_robot_pcd_idx(xyz, bbox_all)
    xyz = xyz[~pcd_idx]
    rgb = rgb[~pcd_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    xyz = torch.tensor(cl.points)

    o3d.visualization.draw_geometries([cl])
    n_xyz = torch.tensor([len(xyz)])
    policy.collision_loss(xyz,)


# test()
