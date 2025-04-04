'''
ForwardKinematics Policy
'''
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
from codebase.z_model.attentionlayer import SelfAttnFFW, CrossAttnFFW
from zero.expAugmentation.ReconLoss.ForwardKinematics import FrankaEmikaPanda
# 先不要参数化
# 先不要大改，按照DP的写法来


'''
Policy
ActionHead, Feature Extractor

'''


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # test
        test1 = torch.arange(half_dim, device=device)
        test2 = test1 * -emb
        test3 = torch.exp(test2)

        # / test
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ---------------------------------------------------------------
# region 0. Some tools


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
        d_model = config['FK']['ActionHead']['d_model']
        d_instr = config['FK']['ActionHead']['d_instr']
        d_ffw = config['FK']['ActionHead']['d_ffw']
        n_heads = config['FK']['ActionHead']['n_heads']
        d_pcd_features = config['FK']['FeatureExtractor']['ptv3']['enc_channels'][-1]

        # ActionEmbedding
        self.embed_actionHis = nn.Linear(8, d_model, bias=False)  # TODO:可能可以离散化，用类似普朗克常量的东西
        self.embed_instr = nn.Linear(d_instr, d_model, bias=False)
        self.embed_t = nn.Embedding(1, d_model)
        self.embed_features = nn.Linear(d_pcd_features, d_model, bias=False)  # TODO:可能可以离散化，用类似普朗克常量的东西
        # 先不对ptv3的feature进行处理

        # position embedding

        self.PE = SinusoidalPosEmb(d_model)

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

        # for encoder
        embeded_pcd = self.embed_features(pcd_feat)
        embeded_actionHis = self.embed_actionHis(JP_hist)
        embeded_instr = self.embed_instr(instr)
        embeded_t = self.embed_t(t).unsqueeze(1)

        # embeded_pcd_empty = torch.ones((len(pcd_feat), n_features, pcd_feat.size(1)), device=pcd_feat.device) * 0.5
        # pcd_feat=
        # embeded_pcd = self.embed_features(embeded_pcd_empty, pcd_feat)
        # for decoder
        embeded_actionFuture = self.embed_actionHis(JP_futr_noisy)

        feature_all = torch.cat([embeded_pcd, embeded_actionHis, embeded_instr, embeded_t], dim=1)
        pe = self.PE(torch.arange(feature_all.size(1), device=feature_all.device))
        feature_all = feature_all + pe
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
        self.ActionHead = ActionHead(config)
        self.FeatureExtractor = FeatureExtractorPTv3Clean(config)
        self.config = config
        self.franka = FrankaEmikaPanda()
        self.action_theta_offset = [0, 0, 0, math.radians(-4), 0, 0, 0]

        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=config['FK']['Policy']['num_timesteps'],
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )

    def forward(self, batch):

        # get data
        JP_futr = batch['JP_futr']  # [batch, horizon, action_dim]
        JP_hist = batch['JP_hist']  # [batch, horizon, action_dim]
        instr = batch['instr']
        # feature extraction
        ptv3_batch = self.FeatureExtractor.prepare_ptv3_batch(batch)
        features = self.FeatureExtractor(ptv3_batch)

        # DDPM scheduler
        noise = torch.randn(JP_futr.size(), device=JP_futr.device)
        t = torch.randint(0, self.ddpm_scheduler.num_train_timesteps, (JP_futr.size(0),), device=JP_futr.device).long()
        JP_futr_noisy = self.ddpm_scheduler.add_noise(JP_futr, noise, timesteps=t)
        # DDOM predict
        JP_t_1 = self.ActionHead(features, JP_hist, instr, t, JP_futr_noisy)

        xyz = batch['pc_fts'][:, :3]
        collision_loss = self.collision_loss(xyz, JP_t_1)
        diffusion_loss = self.diffusion_loss(xyz, JP_t_1)

        loss = collision_loss + diffusion_loss
        return JP_t_1

    def collision_loss(self, xyz, n_xyz, action):
        '''
        Assume the action has shape [B, Horizon,8]
        xyz is a list of Batch, 每个里面有不同的点云,要是点云能大小一致就好了。
        '''

        for i in range(len(xyz)):
            s_action = action[i]  # [Horizon, 8]
            T_ok = torch.tensor([self.franka.get_T_ok(s_action[j])[0] for j in range(s_action.size(0))])  # [Horizon, 7,4,4]
            T_ok = T_ok.unsqueeze(-2)  # [Horizon, 7,1,4,4]

            P_o = xyz[i]  # [N, 3]
            ones = torch.ones((P_o.size(0), 1), device=P_o.device)
            P_o = torch.cat((P_o, ones), dim=1)

            P_op_homo = P_o.unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # [1,1,N,4,1]
            P_kp = torch.matmul(T_ok, P_op_homo)  # [Horizon,7,N,3]

        pass

    def diffusion_loss(self, xyz, action):
        pass
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
