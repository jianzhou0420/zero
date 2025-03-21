from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ...models.lotus.utils.rotation_transform import discrete_euler_to_quaternion
from ...models.lotus.base import BaseModel, RobotPoseEmbedding
from ...models.lotus.utils.rotation_transform import RotationMatrixTransform
from ...models.lotus.PointTransformerV3.model import (
    PointTransformerV3, offset2bincount, offset2batch
)
from ...models.lotus.PointTransformerV3.model_ca import PointTransformerV3CA
from ...models.lotus.utils.action_position_utils import get_best_pos_from_disc_pos

# TODO: horizon


class ActionHead(nn.Module):
    def __init__(
        self, reduce, pos_pred_type, rot_pred_type, hidden_size, dim_actions,
        dropout=0, voxel_size=0.01, euler_resolution=5, ptv3_config=None, pos_bins=50, horizon=8,
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
        self.horizon = horizon

        self.heatmap_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3 * self.pos_bins * 2 * self.horizon)
        )

        output_size = self.euler_bins * 3 + 1

        input_size = hidden_size // self.horizon

        self.action_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, feat, npoints_in_batch):
        '''
        xt: (npoints,H, 3, pos_bins*2)
        xr: (B, H, euler_bins, 3)
        xo: (B, H, 1)
        B: batch_size, H: horizon
        '''

        xt = self.heatmap_mlp(feat)  # (npoints, 3*pos_bins)
        xt = einops.rearrange(xt, 'n (c h b) -> n h c b', c=3, h=self.horizon)  # (3, #npoints, pos_bins)

        # 2. xr
        split_feat = torch.split(feat, npoints_in_batch)  # 按照归属切分 成 64 个tensor，每个tensor(约1050,128)
        # pc_embeds111 = torch.stack([torch.max(x, 0)[0] for x in split_feat], 0)  # 每个tensor是一个点云，
        # 每16个embed

        pc_embeds = []
        for i, each_feature in enumerate(split_feat):
            single_embed = []
            for j in range(self.horizon):
                single_chunk = each_feature[:, j * self.hidden_size // self.horizon: (j + 1) * self.hidden_size // self.horizon]
                single_chunk = torch.max(single_chunk, 0)[0]
                single_embed.append(single_chunk)
            pc_embeds.append(torch.stack(single_embed, 0))
        pc_embeds = torch.stack(pc_embeds, 0)

        action_embeds = self.action_mlp(pc_embeds)

        xr = action_embeds[..., :self.euler_bins * 3].view(-1, self.horizon, self.euler_bins, 3)

        # 3. xo
        xo = action_embeds[..., -1].unsqueeze(-1)

        # xt: (npoints, H, 3, pos_bins * 2)
        # xr: (B, H, euler_bins, 3)
        # xo: (B, H, 1)
        return xt, xr, xo


class MultiActionHead(nn.Module):
    def __init__(
        self, reduce, pos_pred_type, rot_pred_type, hidden_size, dim_actions,
        dropout=0, voxel_size=0.01, euler_resolution=5, ptv3_config=None, pos_bins=50, horizon=8, unit_test=False,
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
        self.horizon = horizon
        self.unit_test = unit_test

        self.heatmap_mlp_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(0.02),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 3 * self.pos_bins * 2)
                ) for _ in range(horizon)
            ]
        )

        output_size = self.euler_bins * 3 + 1

        input_size = hidden_size

        self.action_mlp_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.LeakyReLU(0.02),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size)
                ) for _ in range(horizon)
            ]
        )

    def forward(self, feat, npoints_in_batch):
        '''
        xt: (npoints_all_batch,H, 3, pos_bins*2)
        xr: (B, H, euler_bins, 3)
        xo: (B, H, 1)
        B: batch_size, H: horizon
        '''

        # 1. xt
        if self.unit_test is True:
            torch.manual_seed(42)
            print('multihead is in unit test')
        xt_h = torch.vstack([mlp(feat).unsqueeze(0) for mlp in self.heatmap_mlp_list])  # [h, n, 3 * pos_bins * 2]
        xt_h = einops.rearrange(xt_h, 'h n (c b) -> n h c b', c=3, h=self.horizon)  # (3, #npoints, pos_bins)

        # 2. xr
        split_feat = torch.split(feat, npoints_in_batch)  # 按照归属切分 成 64 个tensor，每个tensor(约1050,128)
        pc_embeds = torch.stack([torch.max(x, 0)[0] for x in split_feat], 0)  # 每个tensor是一个点云，（B，128）

        if self.unit_test is True:
            torch.manual_seed(42)
            # print('pc_embeds', pc_embeds)
        action_embeds = torch.vstack([mlp(pc_embeds).unsqueeze(0) for mlp in self.action_mlp_list])  # [h, B,euler_bins * 3 + 1]
        action_embeds = einops.rearrange(action_embeds, 'h b d -> b h d')  # (B, H, euler_bins * 3 + 1)
        xr_h = action_embeds[..., :self.euler_bins * 3]
        xr_h = einops.rearrange(xr_h, 'b h (d c) -> b h d c', d=self.euler_bins, c=3)

        # 3. xo
        xo_h = action_embeds[..., -1].unsqueeze(-1)  # (B , H,1)
        return xt_h, xr_h, xo_h


class SimplePolicyPTV3AdaNorm(BaseModel):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """

    def __init__(self, config):
        super().__init__()

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

        if config.action_config.action_head_type == 'split':
            self.act_proj_head = ActionHead(
                act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
                config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
                dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
                ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
                euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon
            )
        elif config.action_config.action_head_type == 'multihead':
            self.act_proj_head = MultiActionHead(
                act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
                config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
                dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
                ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
                euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon
            )
        else:
            raise ValueError('Unknown multi_action_head type')

        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()

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
        if self.unit_test is True:
            torch.manual_seed(42)
        point_outs = self.ptv3_model(ptv3_batch, return_dec_layers=True)

        # 2. Action Head
        if self.unit_test is True:
            torch.manual_seed(42)
        pred_actions = self.act_proj_head(point_outs[-1].feat, batch['npoints_in_batch'])

        # 下面关于pred_pos, pred_rot, pred_open的操作在训练时都是无用的
        # 我修改了，只在eval时生效
        # 3.1 get Ground Truth
        if not is_train:  # means eval
            pred_pos, pred_rot, pred_open = pred_actions
            if self.config.action_config.pos_pred_type == 'heatmap_disc':
                # TODO
                # if not compute_loss:
                # import time
                # st = time.time()
                cont_pred_pos = []
                npoints_in_batch = offset2bincount(point_outs[-1].offset).data.cpu().numpy().tolist()
                # (n_points, horizon, channel,pos_bins)
                split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=0)
                split_coords = torch.split(point_outs[-1].coord, npoints_in_batch)
                for i in range(len(npoints_in_batch)):
                    tmp_var = []
                    for j in range(self.config.action_config.horizon):  # TODO: horizon
                        input = split_pred_pos[i][:, j, :, :]
                        input = einops.rearrange(input, 'n c b -> c (n b)')
                        disc_pos_prob = torch.softmax(input, dim=-1)

                        best = get_best_pos_from_disc_pos(
                            disc_pos_prob.data.cpu().numpy(),
                            split_coords[i].data.cpu().numpy(),
                            best=self.config.action_config.get('best_disc_pos', 'max'),
                            topk=split_coords[i].size(1) * 10,
                            pos_bin_size=self.config.action_config.pos_bin_size,
                            pos_bins=self.config.action_config.pos_bins,
                            # best='ens' , topk=1
                        )
                        tmp_var.append(best)
                    cont_pred_pos.append(np.vstack(tmp_var))

                cont_pred_pos = torch.from_numpy(np.array(cont_pred_pos)).float().to(device)
                # print('time', time.time() - st)
                pred_pos = cont_pred_pos

                # 3.2 figure out predicted action type
                # (batch,horizon,euler_bins,3) (1,8,360,3)
                pred_rot = torch.argmax(pred_rot, 2).data.cpu().numpy()

                # (batch, horizon, 3)

                # TODO：这里只是能跑，后面再改
                B, H = pred_rot.shape[0], pred_rot.shape[1]
                pred_rot = np.vstack([discrete_euler_to_quaternion(x, self.act_proj_head.euler_resolution) for x in pred_rot.reshape(-1, 3)])
                pred_rot = pred_rot.reshape(B, H, 4)
                pred_rot = torch.from_numpy(pred_rot).to(device)
                # print('all_shape', pred_pos.shape, pred_rot.shape, pred_open.shape)
                final_pred_actions = torch.cat([pred_pos, pred_rot, pred_open], dim=-1)
                return final_pred_actions

        else:  # if is_train
            losses = self.compute_loss(
                pred_actions, batch['gt_actions'],
                disc_pos_probs=batch.get('disc_pos_probs', None),
                npoints_in_batch=batch['npoints_in_batch']
            )
            return losses

    def compute_loss(self, pred_actions, tgt_actions, disc_pos_probs=None, npoints_in_batch=None):
        """
        Args:
            pred_actions: (batch_size, max_action_len, dim_action)
            tgt_actions: (all_valid_actions, dim_action) / (batch_size, max_action_len, dim_action)
            masks: (batch_size, max_action_len)

            xt: (npoints,H, 3, pos_bins*2)
            xr: (B, H, euler_bins, 3)
            xo: (B, H, 1)

        """

        # 1. get predicted actions and ground truth
        xt_h, xr_h, xo_h = pred_actions
        if self.unit_test is True:
            xt = xt_h.detach().clone()
            xr = xr_h.detach().clone()
            xo = xo_h.detach().clone()
        tgt_pos, tgt_rot, tgt_open = tgt_actions[..., :3], tgt_actions[..., 3:-1], tgt_actions[..., -1]

        # position loss
        # 如果预测的是heatmap，对heatmap和gt的heatmap进行交叉熵, gt的pos的heatmap是已经给出的，放在disc_pos_probs里
        # pos_loss = F.cross_entropy(
        #     pred_pos.view(-1, 100), disc_pos_probs.view(-1, 100), reduction='mean'
        # )
        split_xt_h = torch.split(xt_h, npoints_in_batch, dim=0)  # 这里可以优化

        loss_xt = 0
        for i in range(len(npoints_in_batch)):
            input = einops.rearrange(split_xt_h[i], 'n h c b -> (c h) (n b)')
            target = torch.stack(disc_pos_probs[i], dim=0)
            target = einops.rearrange(target, 'h c (n b) -> (c h) (n b)', b=self.config.action_config.pos_bins * 2)

            if self.unit_test is True and i == 0:
                xt_input = input.detach().clone()
                xt_target = target.detach().clone()
            loss_xt += F.cross_entropy(input, target, reduction='mean')

        loss_xt /= len(npoints_in_batch)  # TODO:horizon

        # xr # cross_entropy should be input (N,C)(batch, classes) target (N)(batch)
        tgt_rot = tgt_rot.long()    # (batch_size,h, 3)

        input = einops.rearrange(xr_h, 'b h euler_bins channel -> (b h) euler_bins channel')  # pred_rot (batch_size,horizon, euler_bins or classes,channel)
        target = einops.rearrange(tgt_rot, 'b h channel -> (b h) channel')  # tgt_rot (batch_size,horizon, 3)
        if self.unit_test is True:
            xr_input = input.detach().clone()
            xr_target = target.detach().clone()
        rot_loss = F.cross_entropy(input, target, reduction='mean')  # TODO: 360shi euler bins

        # xo
        input = einops.rearrange(xo_h.squeeze(-1), 'b h -> (b h)')
        target = einops.rearrange(tgt_open, 'b h -> (b h)')
        if self.unit_test is True:
            xo_input = input.detach().clone()
            xo_target = target.detach().clone()
        open_loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean')

        total_loss = self.config.loss_config.pos_weight * loss_xt + self.config.loss_config.rot_weight * rot_loss + open_loss

        if self.unit_test is True:
            return xt, xr, xo, xt_input, xt_target, xr_input, xr_target, xo_input, xo_target
        return {
            'pos': loss_xt, 'rot': rot_loss, 'open': open_loss,
            'total': total_loss
        }


class SimplePolicyPTV3CA(SimplePolicyPTV3AdaNorm):
    """Cross attention conditioned on text/pose/stepid
    """

    def __init__(self, config, unit_test=False):
        BaseModel.__init__(self)
        self.unit_test = unit_test
        self.config = config
        if self.unit_test is True:
            torch.manual_seed(42)
        self.ptv3_model = PointTransformerV3CA(**config.ptv3_config)

        act_cfg = config.action_config
        self.txt_fc = nn.Linear(act_cfg.txt_ft_size, act_cfg.context_channels)
        if act_cfg.use_ee_pose:
            self.pose_embedding = RobotPoseEmbedding(act_cfg.context_channels)
        if act_cfg.use_step_id:
            self.stepid_embedding = nn.Embedding(act_cfg.max_steps, act_cfg.context_channels)
        if config.action_config.action_head_type == 'split':
            self.act_proj_head = ActionHead(
                act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
                config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
                dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
                ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
                euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon, unit_test=self.unit_test
            )
        elif config.action_config.action_head_type == 'multihead':
            self.act_proj_head = MultiActionHead(
                act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
                config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
                dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
                ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
                euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon, unit_test=self.unit_test
            )
        else:
            raise ValueError('Unknown multi_action_head type')

        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()

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


class SimplePolicyPTV3Concat(SimplePolicyPTV3AdaNorm):
    """Adaptive batch/layer normalization conditioned on text/pose/stepid
    """

    def prepare_ptv3_batch(self, batch):
        outs = {
            'coord': batch['pc_fts'][:, :3],
            'grid_size': self.config.action_config.voxel_size,
            'offset': batch['offset'],
            'batch': offset2batch(batch['offset']),
            'feat': batch['pc_fts'],
        }
        # concatenate context for each point cloud
        ctx_embeds = self.txt_fc(batch['txt_embeds'])

        if self.config.action_config.use_ee_pose:
            pose_embeds = self.pose_embedding(batch['ee_poses'])
            ctx_embeds += pose_embeds

        if self.config.action_config.use_step_id:
            step_embeds = self.stepid_embedding(batch['step_ids'])
            ctx_embeds += step_embeds

        npoints_in_batch = torch.from_numpy(
            np.array(batch['npoints_in_batch'])
        ).to(outs['feat'].device)
        ctx_embeds = torch.repeat_interleave(ctx_embeds, npoints_in_batch, 0)
        outs['feat'] = torch.cat([batch['pc_fts'], ctx_embeds], -1)

        return outs


if __name__ == '__main__':
    from genrobo3d.configs.default import get_config

    config = get_config('genrobo3d/configs/rlbench/simple_policy_ptv3.yaml')
    model = SimplePolicyPTV3AdaNorm(config.MODEL).cuda()
    # model = SimplePolicyPTV3CA(config.MODEL).cuda()

    fake_batch = {
        'pc_fts': torch.rand(100, 6),
        'npoints_in_batch': [30, 70],
        'offset': torch.LongTensor([30, 100]),
        'txt_embeds': torch.rand(2, 1280),
        'txt_lens': [1, 1],
        'ee_poses': torch.rand(2, 8),
        'step_ids': torch.LongTensor([0, 1]),
        'gt_actions': torch.rand(2, 8),
    }

    outs = model(fake_batch, compute_loss=True)
    print(outs[1])
