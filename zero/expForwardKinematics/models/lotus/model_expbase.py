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


class FeatureExtractor(PointTransformerV3):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        ctx_channels=256,
        qkv_bias=True,
        qk_scale=None,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_context_channels=256,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        pdnorm_only_decoder=False,
        add_coords_in_attn=False,
        scaled_cosine_attn=False,  # TODO
    ):
        PointModule.__init__(self)
        # assert enable_flash, 'only implemented flash attention'

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
                context_channels=pdnorm_context_channels,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):  # depth是每个layer的深度
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]): sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        add_coords_in_attn=add_coords_in_attn,
                        qk_norm=qk_norm,
                    ),
                    name=f"block{i}",
                )
                if (not pdnorm_only_decoder) or (s == self.num_stages - 1):
                    enc.add(
                        CABlock(
                            channels=enc_channels[s],
                            num_heads=enc_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]): sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            add_coords_in_attn=add_coords_in_attn,
                            qk_norm=qk_norm,
                        ),
                        name=f"block{i}",
                    )
                    dec.add(
                        CABlock(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            kv_channels=ctx_channels,
                            mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            qk_norm=qk_norm,
                            enable_flash=enable_flash,
                        ),
                        name=f"ca_block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict, return_dec_layers=False):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        # print('before', offset2bincount(point.offset))

        point = self.embedding(point)
        point = self.enc(point)
        # print('after', offset2bincount(point.offset))

        layer_outputs = [self._pack_point_dict(point)]

        if not self.cls_mode:
            if return_dec_layers:
                for i in range(len(self.dec)):
                    for dec_block in self.dec[i]:
                        point = dec_block(point)
                        if type(dec_block) == CABlock:
                            layer_outputs.append(self._pack_point_dict(point))
                return layer_outputs
            else:
                point = self.dec(point)
        return point


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

    def forward(
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
        if self.pos_pred_type.startswith('heatmap_mlp'):
            heatmap_embeds = self.heatmap_mlp(feat)
            if self.pos_pred_type == 'heatmap_mlp3':
                heatmaps = torch.split(heatmap_embeds[:, :3], npoints_in_batch)
                new_coords = coords + heatmap_embeds[:, 3:]
            else:
                heatmaps = torch.split(heatmap_embeds[:, :1], npoints_in_batch)
                new_coords = coords + heatmap_embeds[:, 1:]
            # temp = 0.01
            heatmaps = [torch.softmax(x / temp, dim=0)for x in heatmaps]
            # print([x.sum() for x in heatmaps], [x.size() for x in heatmaps])
            # print(npoints_in_batch, temp, [x.max() for x in heatmaps], [x.min() for x in heatmaps])
            new_coords = torch.split(new_coords, npoints_in_batch)
            if self.pos_pred_type == 'heatmap_mlp3':
                xt = torch.stack([
                    torch.einsum('pc,pc->c', h, p) for h, p in zip(heatmaps, new_coords)
                ], dim=0)
            else:
                xt = torch.stack([
                    torch.einsum('p,pc->c', h.squeeze(1), p) for h, p in zip(heatmaps, new_coords)
                ], dim=0)
            if self.pos_pred_type == 'heatmap_mlp_topk':
                topk = 20  # min(npoints_in_batch)
                topk_idxs = [torch.topk(x[:, 0], topk)[1] for x in heatmaps]
                topk_xt = torch.stack([x[i] for x, i in zip(new_coords, topk_idxs)], 0)
                # topk_xt = new_coords

            # import numpy as np
            # np.save('debug1.npy', {'coords': coords.data.cpu().numpy(), 'new_coords': new_coords[0].data.cpu().numpy(), 'heatmaps': heatmaps[0].data.cpu().numpy()})

        elif self.pos_pred_type == 'heatmap_disc':
            xt = self.heatmap_mlp(feat)  # (npoints, 3*pos_bins)
            xt = einops.rearrange(xt, 'n (c b) -> c n b', c=3)  # (3, #npoints, pos_bins)

        # 2. xr
        if self.reduce == 'max':
            split_feat = torch.split(feat, npoints_in_batch)  # 按照归属切分 成 64 个tensor，每个tensor(约1050,128)

            test0 = split_feat[0]
            test1 = torch.max(split_feat[0], 0)
            test2 = test1[0]
            pc_embeds = torch.stack([torch.max(x, 0)[0] for x in split_feat], 0)  # 每个tensor是一个点云，
            action_embeds = self.action_mlp(pc_embeds)
        elif self.reduce.startswith('multiscale_max'):
            pc_embeds = []
            for dec_layer_embed in dec_layers_embed:
                split_dec_embeds = torch.split(dec_layer_embed.feat, offset2bincount(dec_layer_embed.offset).data.cpu().numpy().tolist())
                pc_embeds.append(
                    F.normalize(torch.stack([torch.max(x, 0)[0] for x in split_dec_embeds], 0), p=2, dim=1)
                )
                # print(torch.stack([torch.max(x, 0)[0] for x in split_dec_embeds], 0).max(), torch.stack([torch.max(x, 0)[0] for x in split_dec_embeds], 0).min())
            pc_embeds = torch.cat(pc_embeds, dim=1)
            action_embeds = self.action_mlp(pc_embeds)
        elif self.reduce == 'mean':
            split_feat = torch.split(feat, npoints_in_batch)
            pc_embeds = torch.stack([torch.mean(x, 0) for x in split_feat], 0)
            action_embeds = self.action_mlp(pc_embeds)
        else:  # attn
            action_embeds = self.action_mlp(feat)
            action_heatmaps = torch.split(action_embeds[:, :1], npoints_in_batch)
            action_heatmaps = [torch.softmax(x / temp, dim=0)for x in action_heatmaps]
            split_action_embeds = torch.split(action_embeds[:, 1:], npoints_in_batch)
            action_embeds = torch.stack([(h * v).sum(dim=0) for h, v in zip(action_heatmaps, split_action_embeds)], 0)

        if self.rot_pred_type == 'quat':
            xr = action_embeds[..., :4]
            xr = xr / xr.square().sum(dim=-1, keepdim=True).sqrt()
        elif self.rot_pred_type == 'rot6d':
            xr = action_embeds[..., :6]
        elif self.rot_pred_type in ['euler', 'euler_delta']:
            xr = action_embeds[..., :3]
        elif self.rot_pred_type == 'euler_disc':
            xr = action_embeds[..., :self.euler_bins * 3].view(-1, self.euler_bins, 3)

        # 3. xo
        xo = action_embeds[..., -1]

        if self.pos_pred_type == 'heatmap_mlp_topk':
            return (xt, topk_xt), xr, xo
        else:
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
            euler_resolution=config.action_config.euler_resolution
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
            pred_pos, pred_rot, pred_open = pred_actions
            if self.config.action_config.pos_pred_type == 'heatmap_disc':
                # TODO
                # if not compute_loss:
                # import time
                # st = time.time()
                cont_pred_pos = []
                npoints_in_batch = offset2bincount(point_outs[-1].offset).data.cpu().numpy().tolist()
                # [(3, npoints, pos_bins)]
                split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=1)
                split_coords = torch.split(point_outs[-1].coord, npoints_in_batch)
                for i in range(len(npoints_in_batch)):
                    disc_pos_prob = torch.softmax(
                        split_pred_pos[i].reshape(3, -1), dim=-1
                    )
                    pred_pos, for_visual = get_best_pos_from_disc_pos(
                        disc_pos_prob.data.cpu().numpy(),
                        split_coords[i].data.cpu().numpy(),
                        best=self.config.action_config.get('best_disc_pos', 'max'),
                        topk=split_coords[i].size(1) * 10,
                        pos_bin_size=self.config.action_config.pos_bin_size,
                        pos_bins=self.config.action_config.pos_bins,
                        # best='ens' , topk=1
                    )
                    cont_pred_pos.append(pred_pos)
                cont_pred_pos = torch.from_numpy(np.array(cont_pred_pos)).float().to(device)
                # print('time', time.time() - st)
                pred_pos = cont_pred_pos

                # 3.2 figure out predicted action type
                if self.config.action_config.rot_pred_type == 'rot6d':
                    # no grad
                    pred_rot = self.rot_transform.matrix_to_quaternion(
                        self.rot_transform.compute_rotation_matrix_from_ortho6d(pred_rot.data.cpu())
                    ).float().to(device)
                elif self.config.action_config.rot_pred_type == 'euler':
                    pred_rot = pred_rot * 180
                    pred_rot = self.rot_transform.euler_to_quaternion(pred_rot.data.cpu()).float().to(device)
                elif self.config.action_config.rot_pred_type == 'euler_delta':
                    pred_rot = pred_rot * 180
                    cur_euler_angles = R.from_quat(batch['ee_poses'][..., 3:7].data.cpu()).as_euler('xyz', degrees=True)
                    pred_rot = pred_rot.data.cpu() + cur_euler_angles
                    pred_rot = self.rot_transform.euler_to_quaternion(pred_rot).float().to(device)
                elif self.config.action_config.rot_pred_type == 'euler_disc':
                    pred_rot = torch.argmax(pred_rot, 1).data.cpu().numpy()
                    pred_rot = np.stack([discrete_euler_to_quaternion(x, self.act_proj_head.euler_resolution) for x in pred_rot], 0)
                    pred_rot = torch.from_numpy(pred_rot).to(device)
                final_pred_actions = torch.cat([pred_pos, pred_rot, pred_open.unsqueeze(-1)], dim=-1)

                return final_pred_actions, for_visual

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
        if self.config.action_config.pos_pred_type == 'heatmap_disc':  # 如果预测的是heatmap，对heatmap和gt的heatmap进行交叉熵, gt的pos的heatmap是已经给出的，放在disc_pos_probs里
            # pos_loss = F.cross_entropy(
            #     pred_pos.view(-1, 100), disc_pos_probs.view(-1, 100), reduction='mean'
            # )
            split_pred_pos = torch.split(pred_pos, npoints_in_batch, dim=1)
            pos_loss = 0
            for i in range(len(npoints_in_batch)):
                pos_loss += F.cross_entropy(
                    split_pred_pos[i].reshape(3, -1), disc_pos_probs[i].to(device), reduction='mean'
                )  # input=(3, npoints*pos_bins), target=(3, npoints*pos_bins)
                # 所以这里学习的是如何从所有bins中选出一个点来。
            pos_loss /= len(npoints_in_batch)
        else:
            pos_loss = F.mse_loss(pred_pos, tgt_pos, reduction='mean')

        # rotation loss
        if self.config.action_config.rot_pred_type == 'quat':
            # Automatically matching the closest quaternions (symmetrical solution)
            tgt_rot_ = -tgt_rot.clone()
            rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
            rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
            select_mask = (rot_loss < rot_loss_).float()
            rot_loss = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()
        elif self.config.action_config.rot_pred_type == 'rot6d':
            tgt_rot6d = self.rot_transform.get_ortho6d_from_rotation_matrix(
                self.rot_transform.quaternion_to_matrix(tgt_rot.data.cpu())
            ).float().to(device)
            rot_loss = F.mse_loss(pred_rot, tgt_rot6d)
        elif self.config.action_config.rot_pred_type == 'euler':
            # Automatically matching the closest angles
            tgt_rot_ = tgt_rot.clone()
            tgt_rot_[tgt_rot < 0] += 2
            tgt_rot_[tgt_rot > 0] -= 2
            rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none')
            rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none')
            select_mask = (rot_loss < rot_loss_).float()
            rot_loss = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()
        elif self.config.action_config.rot_pred_type == 'euler_disc':
            tgt_rot = tgt_rot.long()    # (batch_size, 3)
            rot_loss = F.cross_entropy(pred_rot, tgt_rot, reduction='mean')
        else:  # euler_delta
            rot_loss = F.mse_loss(pred_rot, tgt_rot)

        # openness state loss
        open_loss = F.binary_cross_entropy_with_logits(pred_open, tgt_open, reduction='mean')

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
            euler_resolution=config.action_config.euler_resolution
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
