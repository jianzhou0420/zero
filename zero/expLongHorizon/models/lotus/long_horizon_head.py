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
        xo = action_embeds[..., -1]
        return xt, xr, xo


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

        self.act_proj_head = ActionHead(
            act_cfg.reduce, act_cfg.pos_pred_type, act_cfg.rot_pred_type,
            config.ptv3_config.dec_channels[0], act_cfg.dim_actions,
            dropout=act_cfg.dropout, voxel_size=act_cfg.voxel_size,
            ptv3_config=config.ptv3_config, pos_bins=config.action_config.pos_bins,
            euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon
        )

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
        point_outs = self.ptv3_model(ptv3_batch, return_dec_layers=True)

        # 2. Action Head

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
                # [(3, npoints, pos_bins)]
                split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=0)
                split_coords = torch.split(point_outs[-1].coord, npoints_in_batch)
                for i in range(len(npoints_in_batch)):
                    tmp_var = []
                    for j in range(8):  # TODO: horizon

                        debug0 = split_pred_pos[i][:, j, :, :]
                        disc_pos_prob = torch.softmax(
                            split_pred_pos[i][:, j, :, :].reshape(3, -1), dim=-1
                        )
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

                pred_rot = torch.argmax(pred_rot, 2).data.cpu().numpy()

                # (batch, horizon, 3)

                # TODO：这里只是能跑，后面再改
                B, H = pred_rot.shape[0], pred_rot.shape[1]
                pred_rot = np.vstack([discrete_euler_to_quaternion(x, self.act_proj_head.euler_resolution) for x in pred_rot.reshape(-1, 3)])
                pred_rot = pred_rot.reshape(B, H, 4)
                pred_rot = torch.from_numpy(pred_rot).to(device)
                final_pred_actions = torch.cat([pred_pos, pred_rot, pred_open.unsqueeze(-1)], dim=-1)

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
        """

        # loss_cfg = self.config.loss_config
        device = tgt_actions.device

        # 1. get predicted actions and ground truth
        pred_pos, pred_rot, pred_open = pred_actions
        tgt_pos, tgt_rot, tgt_open = tgt_actions[..., :3], tgt_actions[..., 3:-1], tgt_actions[..., -1]

        # position loss
        # 如果预测的是heatmap，对heatmap和gt的heatmap进行交叉熵, gt的pos的heatmap是已经给出的，放在disc_pos_probs里
        # pos_loss = F.cross_entropy(
        #     pred_pos.view(-1, 100), disc_pos_probs.view(-1, 100), reduction='mean'
        # )
        split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=0)
        pos_loss = 0
        for i in range(len(npoints_in_batch)):
            for j in range(self.config.action_config.horizon):  # 8 horizon
                pos_loss += F.cross_entropy(split_pred_pos[i][:, j, :, :].reshape(3, -1).squeeze(), disc_pos_probs[i][j].to(device), reduction='mean')

        pos_loss /= (len(npoints_in_batch) * self.config.action_config.horizon)  # TODO:horizon

        tgt_rot = tgt_rot.long()    # (batch_size, 3)
        rot_loss = F.cross_entropy(pred_rot.reshape(-1, 360, 3), tgt_rot.reshape(-1, 3), reduction='mean')  # TODO: 360shi euler bins

        # openness state loss
        open_loss = F.binary_cross_entropy_with_logits(pred_open.reshape(-1, 1), tgt_open.reshape(-1, 1), reduction='mean')

        total_loss = self.config.loss_config.pos_weight * pos_loss + \
            self.config.loss_config.rot_weight * rot_loss + open_loss

        return {
            'pos': pos_loss, 'rot': rot_loss, 'open': open_loss,
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
            euler_resolution=config.action_config.euler_resolution, horizon=config.action_config.horizon
        )

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
