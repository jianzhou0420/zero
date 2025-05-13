from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from ..lotus.utils.rotation_transform import discrete_euler_to_quaternion
from ..lotus.base import BaseModel, RobotPoseEmbedding
from ..lotus.utils.rotation_transform import RotationMatrixTransform
from ..lotus.PointTransformerV3.model import (
    PointTransformerV3, offset2bincount, offset2batch
)
from ..lotus.PointTransformerV3.model_ca import PointTransformerV3CA
from ..lotus.utils.action_position_utils import get_best_pos_from_disc_pos


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

        if self.pos_pred_type == 'heatmap_disc':
            self.heatmap_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 3 * self.pos_bins * 2)
            )
        else:
            output_size = 1 + 3
            self.heatmap_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size)
            )

        if self.rot_pred_type == 'euler_disc':
            output_size = self.euler_bins * 3 + 1
        else:
            output_size = dim_actions - 3
        if self.reduce == 'attn':
            output_size += 1
        input_size = hidden_size

        self.action_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, feat, npoints_in_batch, coords=None):
        '''
        Args:
            point_embeds: (# all points, dim)
            npoints_in_batch: (batch_size, )
            coords: (# all points, 3)
        Return:
            pred_actions: (batch, num_steps, dim_actions)
        '''
        # 1. xt

        xt = self.heatmap_mlp(feat)  # (npoints, 3*pos_bins)
        xt = einops.rearrange(xt, 'n (c b) -> c n b', c=3)  # (3, #npoints, pos_bins)

        # 2. xr
        split_feat = torch.split(feat, npoints_in_batch)  # 按照归属切分 成 64 个tensor，每个tensor(约1050,128)
        pc_embeds = torch.stack([torch.max(x, 0)[0] for x in split_feat], 0)  # 每个tensor是一个点云，
        action_embeds = self.action_mlp(pc_embeds)
        xr = action_embeds[..., :self.euler_bins * 3].view(-1, self.euler_bins, 3)

        # 3. xo
        xo = action_embeds[..., -1]

        return xt, xr, xo


class SimplePolicyPTV3CA(BaseModel):
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
            euler_resolution=config.action_config.euler_resolution
        )

        self.apply(self._init_weights)

        self.rot_transform = RotationMatrixTransform()
        print('SimplePolicyPTV3AdaCA')

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
        if not is_train:  # means eval
            self.inference(point_outs, pred_actions, device)
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
        device = tgt_actions.device

        # 1. get predicted actions and ground truth
        pred_pos, pred_rot, pred_open = pred_actions
        tgt_pos, tgt_rot, tgt_open = tgt_actions[..., :3], tgt_actions[..., 3:-1], tgt_actions[..., -1]

        #  xt
        split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=1)
        pos_loss = 0
        for i in range(len(npoints_in_batch)):
            pos_loss += F.cross_entropy(
                split_pred_pos[i].reshape(3, -1), disc_pos_probs[i].to(device), reduction='mean'
            )  # input=(3, npoints*pos_bins), target=(3, npoints*pos_bins)
            # 所以这里学习的是如何从所有bins中选出一个点来。
        pos_loss /= len(npoints_in_batch)

        # xr
        tgt_rot = tgt_rot.long()    # (batch_size, 3)
        rot_loss = F.cross_entropy(pred_rot, tgt_rot, reduction='mean')

        # openness state loss
        open_loss = F.binary_cross_entropy_with_logits(pred_open, tgt_open, reduction='mean')

        total_loss = self.config.loss_config.pos_weight * pos_loss + self.config.loss_config.rot_weight * rot_loss + open_loss

        return {
            'pos': pos_loss, 'rot': rot_loss, 'open': open_loss,
            'total': total_loss
        }

    def inference(self, point_outs, pred_actions, device):
        pred_pos, pred_rot, pred_open = pred_actions
        # xt
        cont_pred_pos = []
        npoints_in_batch = offset2bincount(point_outs[-1].offset).data.cpu().numpy().tolist()
        split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=1)
        split_coords = torch.split(point_outs[-1].coord, npoints_in_batch)

        for i in range(len(npoints_in_batch)):
            disc_pos_prob = torch.softmax(
                split_pred_pos[i].reshape(3, -1), dim=-1
            )
            cont_pred_pos.append(
                get_best_pos_from_disc_pos(
                    disc_pos_prob.data.cpu().numpy(),
                    split_coords[i].data.cpu().numpy(),
                    best=self.config.action_config.get('best_disc_pos', 'max'),
                    topk=split_coords[i].size(1) * 10,
                    pos_bin_size=self.config.action_config.pos_bin_size,
                    pos_bins=self.config.action_config.pos_bins,
                    # best='ens' , topk=1
                )
            )
        cont_pred_pos = torch.from_numpy(np.array(cont_pred_pos)).float().to(device)
        # print('time', time.time() - st)
        pred_pos = cont_pred_pos

        # xr
        pred_rot = torch.argmax(pred_rot, 1).data.cpu().numpy()
        pred_rot = np.stack([discrete_euler_to_quaternion(x, self.act_proj_head.euler_resolution) for x in pred_rot], 0)
        pred_rot = torch.from_numpy(pred_rot).to(device)

        final_pred_actions = torch.cat([pred_pos, pred_rot, pred_open.unsqueeze(-1)], dim=-1)

        return final_pred_actions


if __name__ == '__main__':
    from zero.expForwardKinematics.config.default import get_config
    config_path = './zero/expLongHorizon/config/expBase_Lotus.yaml'

    parameters = [
        "name", "EXP03_02_rollback",
        "dataset", "augment",
        "num_gpus", "1",
        "epoches", "400",
        "batch_size", "4",
        "TRAIN_DATASET.num_points", "100000",
        "TRAIN_DATASET.pos_bins", "75",
        "TRAIN_DATASET.pos_bin_size", "0.001",
        "MODEL.action_config.pos_bins", "75",
        "MODEL.action_config.pos_bin_size", "0.001",
        "horizon", "1",
        "TRAIN_DATASET.pos_heatmap_no_robot", "False",
        "MODEL.action_config.horizon", "1",
        "MODEL.action_config.action_head_type", "multihead",

    ]
    config = get_config(config_path, parameters)

    test = SimplePolicyPTV3CA(config.MODEL)
