from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import matplotlib.pyplot as plt

from zero.expBins.models.lotus.utils.rotation_transform import discrete_euler_to_quaternion
from zero.expBins.models.lotus.base import BaseModel, RobotPoseEmbedding
from zero.expBins.models.lotus.utils.rotation_transform import RotationMatrixTransform
from zero.expBins.models.lotus.PointTransformerV3.model import (
    PointTransformerV3, offset2bincount, offset2batch
)
from zero.expBins.models.lotus.PointTransformerV3.model_ca import PointTransformerV3CA
from zero.expBins.models.lotus.utils.action_position_utils import get_best_pos_from_disc_pos, BIN_SPACE
from zero.expBins.models.lotus.CrossAtten import FeatureAggregate


class ActionHead(nn.Module):
    def __init__(
        self, reduce, pos_pred_type, rot_pred_type, hidden_size, dim_actions,
        dropout=0, voxel_size=0.01, euler_resolution=5, ptv3_config=None, pos_bins=50,
    ) -> None:
        super().__init__()
        assert reduce in ['max', 'mean', 'attn', 'multiscale_max', 'multiscale_max_large']
        assert pos_pred_type in ['heatmap_mlp', 'heatmap_mlp3', 'heatmap_mlp_topk', 'heatmap_mlp_clf', 'heatmap_normmax', 'heatmap_disc']
        assert rot_pred_type in ['quat', 'rot6d', 'euler', 'euler_delta', 'euler_disc']

        self.reduce = reduce
        self.pos_pred_type = pos_pred_type
        self.rot_pred_type = rot_pred_type
        self.hidden_size = hidden_size
        self.dim_actions = dim_actions
        self.voxel_size = voxel_size
        self.euler_resolution = euler_resolution
        self.euler_bins = 360 // euler_resolution
        self.pos_bins = pos_bins

        x_bins_location = np.arange(BIN_SPACE[0][0], BIN_SPACE[0][1], 0.001)  # TODO: 0.001
        y_bins_location = np.arange(BIN_SPACE[1][0], BIN_SPACE[1][1], 0.001)
        z_bins_location = np.arange(BIN_SPACE[2][0], BIN_SPACE[2][1], 0.001)
        self.bins_locations = [x_bins_location, y_bins_location, z_bins_location]

        num_queries = len(x_bins_location) + len(y_bins_location) + len(z_bins_location)

        ft_ag_dim = hidden_size / 3
        self.xt_cross_atten = FeatureAggregate(num_queries=num_queries, feature_dim=hidden_size)
        self.xr_cross_atten = FeatureAggregate(num_queries=72 * 3, feature_dim=hidden_size)
        self.xo_cross_atten = FeatureAggregate(num_queries=1, feature_dim=hidden_size)

        # self.xt_mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(0.02),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 1)
        # )
        # self.xr_mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(0.02),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 1)
        # )
        # self.xo_mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(0.02),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, 1)
        # )

    def forward(  # TODO： 重写
        self, feat, npoints_in_batch, coords=None, temp=1,
        gt_pos=None, dec_layers_embed=None,
    ):
        '''
        Args:
            point_embeds: (# all points, dim)
            npoints_in_batch: (batch_size, )
            coords: (# all points, 3)
        Return:
            pred_actions: (batch, num_steps, dim_actions)
        '''
        # 1. xt
        feat_all = torch.split(feat, npoints_in_batch, dim=0)

        xt = []
        xr = []
        xo = []
        for feat in feat_all:
            xt_feat = self.xt_cross_atten(feat)  # [num_bins, hidden_size]
            xr_feat = self.xr_cross_atten(feat)  # [72 * 3, hidden_size]
            xo_feat = self.xo_cross_atten(feat)  # [1, hidden_size]

            # xt_mlp = self.xt_mlp(xt_feat)  # [num_bins, 1]
            # xr_mlp = self.xr_mlp(xr_feat)  # [72 * 3, 1]
            # xo_mlp = self.xo_mlp(xo_feat)  # [1, 1]
            xt_feat = torch.sum(xt_feat, dim=1)
            xr_feat = torch.sum(xr_feat, dim=1)
            xo_feat = torch.sum(xo_feat, dim=1)

            xt.append(xt_feat)  # [num_bins, 1] prob of each bin
            xr.append(xr_feat)
            xo.append(xo_feat)

        return xt, xr, xo


class SimplePolicyPTV3AdaNorm(BaseModel):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """

    def __init__(self, config):
        super().__init__()

        config.defrost()
        config.ptv3_config.pdnorm_only_decoder = config.ptv3_config.get('pdnorm_only_decoder', False)
        config.freeze()

        self.config = config

        self.ptv3_model = PointTransformerV3(**config.ptv3_config)

        act_cfg = config.action_config
        self.txt_fc = nn.Linear(act_cfg.txt_ft_size, act_cfg.context_channels)
        if act_cfg.txt_reduce == 'attn':
            self.txt_attn_fc = nn.Linear(act_cfg.txt_ft_size, 1)
        if act_cfg.use_ee_pose:
            self.pose_embedding = RobotPoseEmbedding(act_cfg.context_channels)
        if act_cfg.use_step_id:
            self.stepid_embedding = nn.Embedding(act_cfg.max_steps, act_cfg.context_channels)

        self.act_proj_head = ActionHead(
            act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
            config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
            dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
            ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
        )

        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()

        # debug

    def prepare_ptv3_batch(self, batch):
        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config.action_config.voxel_size,
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }
        # encode context for each point cloud
        ctx_embeds = self.txt_fc(batch['txt_embeds'])
        if self.config.action_config.txt_reduce == 'attn':
            txt_weights = torch.split(self.txt_attn_fc(batch['txt_embeds']), batch['txt_lens'])
            txt_embeds = torch.split(ctx_embeds, batch['txt_lens'])
            ctx_embeds = []
            for txt_weight, txt_embed in zip(txt_weights, txt_embeds):
                txt_weight = torch.softmax(txt_weight, 0)
                ctx_embeds.append(torch.sum(txt_weight * txt_embed, 0))
            ctx_embeds = torch.stack(ctx_embeds, 0)

        if self.config.action_config.use_ee_pose:
            pose_embeds = self.pose_embedding(batch['ee_poses'])
            ctx_embeds += pose_embeds

        if self.config.action_config.use_step_id:
            step_embeds = self.stepid_embedding(batch['step_ids'])
            ctx_embeds += step_embeds

        outs['context'] = ctx_embeds

        return outs

    def forward(self, batch, is_train=False):
        '''batch data:
            pc_fts: (batch, npoints, dim)
            txt_embeds: (batch, txt_dim)
        '''
        batch = self.prepare_batch(batch)  # send to device
        device = batch['pc_fts'].device

        ptv3_batch = self.prepare_ptv3_batch(batch)

        # 1. Point Transformer V3
        point_outs = self.ptv3_model(ptv3_batch, return_dec_layers=True)

        # 2. Action Head
        pred_actions = self.act_proj_head(  # 只用到了point_outs的最后一层
            point_outs[-1].feat, batch['npoints_in_batch'], coords=point_outs[-1].coord,
            temp=self.config.action_config.get('pos_heatmap_temp', 1),
            gt_pos=batch['gt_actions'][..., :3] if 'gt_actions' in batch else None,
            # dec_layers_embed=[point_outs[k] for k in [0, 2, 4, 6, 8]] # TODO
            dec_layers_embed=[point_outs[k] for k in [0, 1, 2, 3, 4]] if self.config.ptv3_config.dec_depths[0] == 1 else [point_outs[k] for k in [0, 2, 4, 6, 8]]  # TODO
        )
        # 下面关于pred_pos, pred_rot, pred_open的操作在训练时都是无用的
        # 我修改了，只在eval时生效
        # 3.1 get Ground Truth
        if is_train:
            losses = self.compute_loss(
                pred_actions, batch['gt_actions'],
                disc_pos_probs=batch.get('disc_pos_probs', None),
                npoints_in_batch=batch['npoints_in_batch']
            )
            return losses
        else:
            xt, xr, xo = pred_actions
            bs = len(xt)
            pred_pos = []
            for i in range(bs):
                # 找出第几个bin是我们需要的
                n_bins_x, n_bins_y, n_bins_z = [len(bins) for bins in self.act_proj_head.bins_locations]
                bin_x = xt[i][:n_bins_x]
                bin_y = xt[i][n_bins_x: n_bins_x + n_bins_y]
                bin_z = xt[i][n_bins_x + n_bins_y:]

                x_idx = torch.argmax(bin_x)
                y_idx = torch.argmax(bin_y)
                z_idx = torch.argmax(bin_z)
                x_pos = self.act_proj_head.bins_locations[0][x_idx]
                y_pos = self.act_proj_head.bins_locations[1][y_idx]
                z_pos = self.act_proj_head.bins_locations[2][z_idx]
                # plt.ion()
                # plt.plot(bin_x.cpu().detach().numpy())
                # plt.show()
                # print idx
                # print(f'x: {x_idx}, y: {y_idx}, z: {z_idx}')
                pred_pos.append(torch.from_numpy(np.array([x_pos, y_pos, z_pos])).to('cuda'))
            # xr
            pred_rot = torch.argmax(xr[0].unsqueeze(0), 1).data.cpu().numpy()
            pred_rot = np.stack([discrete_euler_to_quaternion(x, self.act_proj_head.euler_resolution) for x in pred_rot], 0)
            pred_rot = torch.from_numpy(pred_rot).to(device)

            pred_open = xo
            final_pred_actions = torch.cat([pred_pos[0].unsqueeze(0), pred_rot, pred_open], dim=-1)  # TODO: list fix
            return final_pred_actions

    def compute_loss(self, pred_actions, tgt_actions, disc_pos_probs=None, npoints_in_batch=None):
        """
        Args:
            pred_actions: (batch_size, max_action_len, dim_action)
            tgt_actions: (all_valid_actions, dim_action) / (batch_size, max_action_len, dim_action)
            masks: (batch_size, max_action_len)
        """
        device = pred_actions[0][0].device
        # loss_cfg = self.config.loss_config
        xt, xr, xo = pred_actions
        tgt_t, tgt_r, tgt_o = tgt_actions[..., :3], tgt_actions[..., 3:-1], tgt_actions[..., -1]
        # xt loss

        # bins_num
        n_bins_x, n_bins_y, n_bins_z = [len(bins) for bins in self.act_proj_head.bins_locations]

        xt_loss = 0
        for i in range(len(npoints_in_batch)):

            xt_x_loss = F.cross_entropy(xt[i][:n_bins_x].squeeze(), disc_pos_probs[i][:n_bins_x].to(device), reduction='mean')
            xt_y_loss = F.cross_entropy(xt[i][n_bins_x: n_bins_x + n_bins_y].squeeze(), disc_pos_probs[i][n_bins_x: n_bins_x + n_bins_y].to(device), reduction='mean')
            xt_z_loss = F.cross_entropy(xt[i][n_bins_x + n_bins_y:].squeeze(), disc_pos_probs[i][n_bins_x + n_bins_y:].to(device), reduction='mean')
            xt_loss += (xt_x_loss + xt_y_loss + xt_z_loss) / 3

        xt_loss /= len(npoints_in_batch)

        # xr loss
        tgt_r = tgt_r.long()  # (bs, 3)
        xr = torch.stack(xr, 0).view(-1, 72, 3)
        xr_loss = F.cross_entropy(xr, tgt_r, reduction='mean')

        # xo loss
        xo = torch.stack(xo, 0).squeeze()
        xo_loss = F.binary_cross_entropy_with_logits(xo, tgt_o, reduction='mean')
        total_loss = self.config.loss_config.pos_weight * xt_loss + \
            self.config.loss_config.rot_weight * xr_loss + xo_loss

        if self.print_flag:
            print(f'xt_loss: {xt_loss.item()}, xr_loss: {xr_loss.item()}, xo_loss: {xo_loss.item()}')
            self.print_flag = False
        return {
            'pos': xt_loss, 'rot': xr_loss, 'open': xo_loss,
            'total': total_loss
        }


class SimplePolicyPTV3CA(SimplePolicyPTV3AdaNorm):
    """Cross attention conditioned on text/pose/stepid
    """

    def __init__(self, config):
        BaseModel.__init__(self)

        self.config = config

        self.ptv3_model = PointTransformerV3CA(**config.ptv3_config)

        act_cfg = config.action_config
        self.txt_fc = nn.Linear(act_cfg.txt_ft_size, act_cfg.context_channels)
        if act_cfg.use_ee_pose:
            self.pose_embedding = RobotPoseEmbedding(act_cfg.context_channels)
        if act_cfg.use_step_id:
            self.stepid_embedding = nn.Embedding(act_cfg.max_steps, act_cfg.context_channels)
        self.act_proj_head = ActionHead(
            act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
            config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
            dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
            ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
        )

        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()
        self.print_flag = True

    def prepare_ptv3_batch(self, batch):
        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config.action_config.voxel_size,
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }
        device = batch['pc_fts'].device

        # encode context for each point cloud
        txt_embeds = self.txt_fc(batch['txt_embeds'])
        ctx_embeds = torch.split(txt_embeds, batch['txt_lens'])
        ctx_lens = torch.LongTensor(batch['txt_lens'])

        if self.config.action_config.use_ee_pose:
            pose_embeds = self.pose_embedding(batch['ee_poses'])
            ctx_embeds = [torch.cat([c, e.unsqueeze(0)], dim=0) for c, e in zip(ctx_embeds, pose_embeds)]
            ctx_lens += 1

        if self.config.action_config.use_step_id:
            step_embeds = self.stepid_embedding(batch['step_ids'])
            ctx_embeds = [torch.cat([c, e.unsqueeze(0)], dim=0) for c, e in zip(ctx_embeds, step_embeds)]
            ctx_lens += 1

        outs['context'] = torch.cat(ctx_embeds, 0)
        outs['context_offset'] = torch.cumsum(ctx_lens, dim=0).to(device)

        return outs
