'''
terminology declaration:



'''
import einops
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from zero.expForwardKinematics.models.Base.BaseAll import BasePolicy
from zero.z_utils.utilities_all import normalize_JP, denormalize_JP, normalize_pos
import torch.nn.functional as F
from zero.z_utils.DA3D import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
from torchvision.ops import FeaturePyramidNetwork
from zero.expForwardKinematics.models.DiffuserActor3D.components.position_encodings import RotaryPositionEncoding3D
from zero.expForwardKinematics.models.DiffuserActor3D.components.layers import FFWRelativeCrossAttentionModule, ParallelAttention
from zero.expForwardKinematics.models.DiffuserActor3D.components.resnet import load_resnet50, load_resnet18
from zero.expForwardKinematics.models.DiffuserActor3D.components.clip import load_clip


import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from zero.expForwardKinematics.models.DiffuserActor3D.components.position_encodings import SinusoidalPosEmb, RotaryPositionEncoding3D
from zero.expForwardKinematics.models.DiffuserActor3D.components.layers import (
    ParallelAttention,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfAttentionModule,
    FFWRelativeSelfCrossAttentionModule)
from zero.expForwardKinematics.models.Base.BaseAll import BaseActionHead

from zero.expForwardKinematics.config.default import get_config
import dgl.geometry as dgl_geo
# ------------------------------------------------
# region FeatureExtractor


class FeatureExtractor(nn.Module):
    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_sampling_level=3,
                 nhistory=8,
                 nfuture=8,
                 num_attn_heads=8,
                 num_vis_ins_attn_layers=2,
                 fps_subsampling_factor=5,
                 use_instruction=True
                 ):
        super().__init__()
        assert backbone in ["resnet50", "resnet18", "clip"]
        assert image_size in [[128, 128], [256, 256]]
        assert num_sampling_level in [1, 2, 3, 4]

        self.image_size = image_size
        self.num_sampling_level = num_sampling_level
        self.fps_subsampling_factor = fps_subsampling_factor
        self.use_instr = use_instruction
        # Frozen backbone
        if backbone == "resnet50":
            self.backbone, self.normalize = load_resnet50()
        elif backbone == "resnet18":
            self.backbone, self.normalize = load_resnet18()
        elif backbone == "clip":
            self.backbone, self.normalize = load_clip()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Semantic visual features at different scales
        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], embedding_dim
        )
        if self.image_size == [128, 128]:
            # Coarse RGB features are the 2nd layer of the feature pyramid
            # at 1/4 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (64x64)
            self.coarse_feature_map = ['res2', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [4, 2, 2, 2]
        elif self.image_size == [256, 256]:
            # Coarse RGB features are the 3rd layer of the feature pyramid
            # at 1/8 resolution (32x32)
            # Fine RGB features are the 1st layer of the feature pyramid
            # at 1/2 resolution (128x128)
            self.feature_map_pyramid = ['res3', 'res1', 'res1', 'res1']
            self.downscaling_factor_pyramid = [8, 2, 2, 2]

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Current gripper learnable features
        self.action_hist_embed = nn.Embedding(nhistory, embedding_dim)

        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        )

        # Goal gripper learnable features
        self.action_future_embed = nn.Embedding(nfuture, embedding_dim)

        # instr encoder
        self.instr_encoder = nn.Linear(512, embedding_dim)

        # Attention from vision to language
        layer = ParallelAttention(
            num_layers=num_vis_ins_attn_layers,
            d_model=embedding_dim, n_heads=num_attn_heads,
            self_attention1=False, self_attention2=False,
            cross_attention1=True, cross_attention2=False
        )
        self.vl_attention = nn.ModuleList([
            layer
            for _ in range(1)
            for _ in range(1)
        ])

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, nhist, 3+)

        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        return self._encode_gripper(curr_gripper, self.action_hist_embed,
                                    context_feats, context)

    def encode_goal_gripper(self, goal_gripper, context_feats, context):
        """
        Compute goal gripper position features and positional embeddings.

        Args:
            - goal_gripper: (B, 3+)

        Returns:
            - goal_gripper_feats: (B, 1, F)
            - goal_gripper_pos: (B, 1, F, 2)
        """
        goal_gripper_feats, goal_gripper_pos = self._encode_gripper(
            goal_gripper[:, None], self.action_future_embed,
            context_feats, context
        )
        return goal_gripper_feats, goal_gripper_pos

    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        # Learnable embedding for gripper
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )

        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos

    def encode_images(self, rgb, pcd):
        """
        Compute visual features/pos embeddings at different scales.

        Args:
            - rgb: (B, ncam, 3, H, W), pixel intensities
            - pcd: (B, ncam, 3, H, W), positions

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        num_cameras = rgb.shape[1]

        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_features = self.backbone(rgb)

        # Pass visual features through feature pyramid network
        rgb_features = self.feature_pyramid(rgb_features)

        # Treat different cameras separately
        pcd = einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w")

        rgb_feats_pyramid = []
        pcd_pyramid = []
        for i in range(self.num_sampling_level):
            # Isolate level's visual features
            rgb_features_i = rgb_features[self.feature_map_pyramid[i]]

            # Interpolate xy-depth to get the locations for this level
            feat_h, feat_w = rgb_features_i.shape[-2:]
            pcd_i = F.interpolate(
                pcd,
                (feat_h, feat_w),
                mode='bilinear'
            )

            # Merge different cameras for clouds, separate for rgb features
            h, w = pcd_i.shape[-2:]
            pcd_i = einops.rearrange(
                pcd_i,
                "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
            )
            rgb_features_i = einops.rearrange(
                rgb_features_i,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )

            rgb_feats_pyramid.append(rgb_features_i)
            pcd_pyramid.append(pcd_i)
        return rgb_feats_pyramid, pcd_pyramid

    def encode_instr(self, instr):
        """
        Compute language features/pos embeddings on top of CLIP features.

        Args:
            - instr: (B, max_instr_length, 512)

        Returns:
            - instr_feats: (B, 53, F)
            - instr_dummy_pos: (B, 53, F, 2)
        """
        instr_feats = self.instr_encoder(instr)
        # Dummy positional embeddings, all 0s
        instr_dummy_pos = torch.zeros(
            len(instr), instr_feats.shape[1], 3,
            device=instr.device
        )
        instr_dummy_pos = self.relative_pe_layer(instr_dummy_pos)
        return instr_feats, instr_dummy_pos

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos

    def vision_language_attention(self, feats, instr_feats):
        feats, _ = self.vl_attention[0](
            seq1=feats, seq1_key_padding_mask=None,
            seq2=instr_feats, seq2_key_padding_mask=None,
            seq1_pos=None, seq2_pos=None,
            seq1_sem_pos=None, seq2_sem_pos=None
        )
        return feats

    def forward(self, rgb, pcd, instr, action):  # 0418: forward一点问题没有，encode somthing是基本直接抄的，也没问题。
        '''
        Arguments:
            rgb: (B, ncam, 3, H, W)
            pcd: (B, ncam, 3, H, W)
            instr: (B, max_instr_length, 512)
            action: (B, horizon, njoint+open)
        '''

        # normalize joint position
        # joint_position = normalize_theta_positions(action)

        # extract features
        rgb_feats_pyramid, pcd_pyramid = self.encode_images(rgb, pcd)
        context_feats = einops.rearrange(rgb_feats_pyramid[0], "b ncam c h w -> b (ncam h w) c")
        context = pcd_pyramid[0]

        # encode instr

        instr_feats = None
        if self.use_instr:
            instr_feats, _ = self.encode_instr(instr)
            # Attention from vision to language
            context_feats = self.vision_language_attention(
                context_feats, instr_feats
            )

        # encode action
        his_action_feats, _ = self._encode_gripper(action, self.action_hist_embed, context_feats, context)

        # run fps

        fps_feats, fps_pos = self.run_fps(context_feats.transpose(0, 1), self.relative_pe_layer(context))

        return context_feats, context, instr_feats, his_action_feats, fps_feats, fps_pos
# endregion
# ------------------------------------------------
# region ActionHead


class ActionHead_JP(BaseActionHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instr=False,
                 nhist=3,
                 lang_enhanced=False,
                 action_dim=7,  # JP是7
                 horizon=8,
                 ):

        super().__init__()
        self.use_instr = use_instr
        self.lang_enhanced = lang_enhanced
        rotation_dim = 6
        # embedding
        self.traj_embedding = nn.Linear(7, embedding_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.A_hist = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.traj_time_embedding = SinusoidalPosEmb(embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )

        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 7)  # minus 1 for openess
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def encode_denoising_timestep(self, timestep, hist_actions_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_embedding(timestep)

        hist_actions_features = einops.rearrange(
            hist_actions_features, "npts b c -> b npts c"
        )
        hist_actions_features = hist_actions_features.flatten(1)
        hist_action_emb = self.A_hist(hist_actions_features)

        return time_feats + hist_action_emb

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):

        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def prediction_head(self,
                        action_pcd, action_noisy,
                        pcd_pyrimid, pcd_pyrimid_feats,
                        timesteps, curr_gripper_features,
                        fps_pcd_feats, fps_pcd,
                        instr_feats):  # 0418 这里和DA3D几乎一模一样。
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instr_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(action_pcd)
        rel_context_pos = self.relative_pe_layer(pcd_pyrimid)

        # Cross attention from gripper to full context
        action_noisy = self.cross_attn(
            query=action_noisy,
            value=pcd_pyrimid_feats,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([action_noisy, fps_pcd_feats], 0)
        rel_pos = torch.cat([rel_gripper_pos, fps_pcd], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = action_noisy.shape[0]

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        openess = self.openess_predictor(position_features)
        return position, openess

    def forward(self, noisy_traj, timestep, features_all):  # 0418 这里和DA3D几乎一模一样。
        """
        Arguments:
            trajectory: (B, H, 8)
            timestep: (B, 1)
            context_feats: (B, N, dim)
            context: (B, N, dim, 2)
            instr_feats: (B, max_instr_length, dim)
            adaln_gripper_feats: (B, nhist, dim)
            fps_feats: (N, B, dim), N < context_feats.size(1)
            fps_pos: (B, N, dim, 2)
        """
        # Trajectory features
        pcd_feats, context, instr_feats, hist_action_feats, fps_feats, fps_pos = features_all

        horizon = noisy_traj.size(1)
        traj_feats = self.traj_embedding(noisy_traj)  # (B, L, dim)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_embedding(torch.arange(0, horizon, device=traj_feats.device))[None].repeat(len(traj_feats), 1, 1)

        if self.use_instr:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        pcd_feats = einops.rearrange(pcd_feats, 'b l c -> l b c')
        hist_action_feats = einops.rearrange(
            hist_action_feats, 'b l c -> l b c'
        )
        pos_pred, openess_pred = self.prediction_head(
            noisy_traj[..., :3], traj_feats,
            context[..., :3], pcd_feats,
            timestep, hist_action_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        return [torch.cat((pos_pred, openess_pred), -1)]


class ActionHead_eePose(BaseActionHead):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instr=False,
                 nhist=3,
                 lang_enhanced=False,
                 action_dim=9,
                 horizon=8,
                 ):

        super().__init__()
        self.use_instr = use_instr
        self.lang_enhanced = lang_enhanced
        rotation_dim = 6
        # embedding
        self.traj_embedding = nn.Linear(action_dim, embedding_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.A_hist = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.traj_time_embedding = SinusoidalPosEmb(embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )

        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)  # minus 1 for openess
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def encode_denoising_timestep(self, timestep, hist_actions_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_embedding(timestep)

        hist_actions_features = einops.rearrange(
            hist_actions_features, "npts b c -> b npts c"
        )
        hist_actions_features = hist_actions_features.flatten(1)
        hist_action_emb = self.A_hist(hist_actions_features)

        return time_feats + hist_action_emb

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def prediction_head(self,
                        action_pcd, action_noisy,
                        pcd_pyrimid, pcd_pyrimid_feats,
                        timesteps, curr_gripper_features,
                        fps_pcd_feats, fps_pcd,
                        instr_feats):  # 0418 这里和DA3D几乎一模一样。
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instr_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(action_pcd)
        rel_context_pos = self.relative_pe_layer(pcd_pyrimid)

        # Cross attention from gripper to full context
        action_noisy = self.cross_attn(
            query=action_noisy,
            value=pcd_pyrimid_feats,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([action_noisy, fps_pcd_feats], 0)
        rel_pos = torch.cat([rel_gripper_pos, fps_pcd], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = action_noisy.shape[0]

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )
        rotation = self.predict_rot(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )
        openess = self.openess_predictor(position_features)
        return position, rotation, openess

    def forward(self, noisy_traj, timestep, features_all):  # 0418 这里和DA3D几乎一模一样。
        """
        Arguments:
            trajectory: (B, H, 8)
            timestep: (B, 1)
            context_feats: (B, N, dim)
            context: (B, N, dim, 2)
            instr_feats: (B, max_instr_length, dim)
            adaln_gripper_feats: (B, nhist, dim)
            fps_feats: (N, B, dim), N < context_feats.size(1)
            fps_pos: (B, N, dim, 2)
        """
        # Trajectory features
        pcd_feats, context, instr_feats, hist_action_feats, fps_feats, fps_pos = features_all

        horizon = noisy_traj.size(1)
        traj_feats = self.traj_embedding(noisy_traj)  # (B, L, dim)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_embedding(torch.arange(0, horizon, device=traj_feats.device))[None].repeat(len(traj_feats), 1, 1)

        if self.use_instr:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        pcd_feats = einops.rearrange(pcd_feats, 'b l c -> l b c')
        hist_action_feats = einops.rearrange(
            hist_action_feats, 'b l c -> l b c'
        )
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            noisy_traj[..., :3], traj_feats,
            context[..., :3], pcd_feats,
            timestep, hist_action_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]


# endregion
# ------------------------------------------------
# region Policy


class Policy(BasePolicy):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(**config['DiffuserActor']['FeatureExtractor'])
        self.action_space = config['DiffuserActor']['Policy']['action_space']
        self.max_t = config['DiffuserActor']['Policy']['num_timesteps']
        if self.action_space == 'JP':
            self.action_head = ActionHead_JP(**config['DiffuserActor']['ActionHead'])
            self.JP_scheduler = DDPMScheduler(
                num_train_timesteps=config['DiffuserActor']['Policy']['num_timesteps'],
                beta_schedule="scaled_linear",
                prediction_type="epsilon"
            )
        elif self.action_space == 'eePose':
            self.action_head = ActionHead_eePose(**config['DiffuserActor']['ActionHead'])
            self.pos_scheduler = DDPMScheduler(
                num_train_timesteps=config['DiffuserActor']['Policy']['num_timesteps'],
                beta_schedule="scaled_linear",
                prediction_type="epsilon"
            )
            self.rot_scheduler = DDPMScheduler(
                num_train_timesteps=config['DiffuserActor']['Policy']['num_timesteps'],
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon"
            )

    def inference_one_sample_JP(self, batch):
        rgb = batch['rgb']
        pcd = batch['xyz']
        instr = batch['instr']
        A_hist = batch['JP_hist']

        # Condition on start-end pose
        B, nhist, D = A_hist.shape
        condition_data = torch.zeros(
            (B, 8, D),  # 8是horizon
            device=rgb.device
        )
        cond_mask = torch.zeros_like(condition_data)
        cond_mask = cond_mask.bool()
        # feature extraction
        features_all = self.feature_extractor(rgb, pcd, instr, A_hist)
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.JP_scheduler.timesteps[0])

        noisy_JP = self.JP_scheduler.add_noise(
            condition_data, noise, noise_t
        )

        timesteps = self.JP_scheduler.timesteps
        for t in timesteps:
            out = self.action_head(noisy_JP[..., :7], t * torch.ones(len(noisy_JP)).to(noisy_JP.device).long(), features_all)
            noisy_JP = self.JP_scheduler.step(out[0][..., :7], t, noisy_JP[..., :7]).prev_sample
        JP_0 = torch.cat((noisy_JP, out[0][..., 7:]), dim=-1)
        return JP_0

    def inference_one_sample_eePose(self, batch):
        rgb = batch['rgb']
        pcd = batch['xyz']
        instr = batch['instr']
        A_hist = batch['eePose_hist']

        # Condition on start-end pose
        B, nhist, D = A_hist.shape
        H = self.config['DiffuserActor']['ActionHead']['horizon']
        condition_data = torch.zeros(
            (B, H, D),  # 8是horizon
            device=rgb.device
        )
        cond_mask = torch.zeros_like(condition_data)
        cond_mask = cond_mask.bool()
        # feature extraction
        features_all = self.feature_extractor(rgb, pcd, instr, A_hist)
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.pos_scheduler.timesteps[0])

        pos = self.pos_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        rot = self.rot_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_eePose = torch.cat((pos, rot), dim=-1)

        timesteps = self.pos_scheduler.timesteps
        for t in timesteps:
            out = self.action_head(
                noisy_eePose,
                t * torch.ones(len(noisy_eePose)).to(noisy_eePose.device).long(),
                features_all
            )
            out = out[-1]  # keep only last layer's output
            pos = self.pos_scheduler.step(
                out[..., :3], t, noisy_eePose[..., :3]
            ).prev_sample
            rot = self.rot_scheduler.step(
                out[..., 3:9], t, noisy_eePose[..., 3:9]
            ).prev_sample
            noisy_eePose = torch.cat((pos, rot), -1)

        JP_0 = torch.cat((noisy_eePose, out[..., 9:]), dim=-1)
        return JP_0

    def forward(self, batch):
        '''
        Arguments:
            batch:{
                'rgb': torch.Tensor (B, ncam,3, H, W)
                'xyz': torch.Tensor (B, ncam,3, H, W)
                'A_futr': torch.Tensor (B, horizon, njoint+open) 第一帧是current position,最后一帧是下一个keypose的position
                'A_hist': torch.Tensor (B, history, njoint+open) 最后一帧是current position
                'instr': torch.Tensor (B, max_instr_length, dim)
            }

        Returns:

        Note:
            joint_position should be already normalized
        '''
        rgb = batch['rgb']
        pcd = batch['xyz']
        instr = batch['instr']
        if self.action_space == 'JP':
            A_hist = batch['JP_hist']
            A_futr = batch['JP_futr']
        elif self.action_space == 'eePose':
            A_hist = batch['eePose_hist']
            A_futr = batch['eePose_futr']
        else:
            raise NotImplementedError(f"Action space {self.action_space} not implemented")

        # feature extraction
        features_all = self.feature_extractor(rgb, pcd, instr, A_hist)

        # diffusion stuff
        noise = torch.randn(A_futr.shape, device=A_futr.device)
        timesteps = torch.randint(0, self.max_t, (len(noise),), device=noise.device).long()

        if self.action_space == 'eePose':
            noisy_pos = self.pos_scheduler.add_noise(A_futr[..., :3], noise[..., :3], timesteps)
            noisy_rot = self.pos_scheduler.add_noise(A_futr[..., 3:9], noise[..., 3:9], timesteps)
            A_futr_noisy = torch.cat((noisy_pos, noisy_rot), dim=-1)
        elif self.action_space == 'JP':
            A_futr_noisy = self.JP_scheduler.add_noise(A_futr[..., :7], noise[..., :7], timesteps)

        pred = self.action_head(A_futr_noisy, timesteps, features_all)

        # loss
        joint_positon_loss = F.mse_loss(pred[0], A_futr)
        return joint_positon_loss


# endregion


def test_policy():
    import pickle
    example_data_path = './1_Data/B_Preprocess/DA3D/close_jar/variation0/episodes/episode0/data.pkl'
    config_path = './zero/expForwardKinematics/config/DA3D.yaml'
    with open(example_data_path, 'rb') as f:
        example_data = pickle.load(f)

    config = get_config(config_path)
    policy = Policy(config)

    batch = {
        'rgb': torch.randn(2, 2, 3, 256, 256),
        'pcd': torch.randn(2, 2, 3, 256, 256),
        'A_futr': torch.randn(2, 8, 8),
        'A_hist': torch.randn(2, 8, 8),
        'timestep': torch.randint(0, 100, (2,)),
        'instr': torch.randn(2, 77, 512)
    }
    loss = policy(batch)
    print(loss)


if __name__ == '__main__':
    test_policy()
# test_policy()
