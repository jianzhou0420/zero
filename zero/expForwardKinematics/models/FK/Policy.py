'''
ForwardKinematics Policy
'''
import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import einops
import torch.nn.functional as F
# PointTransformer V3
from zero.expForwardKinematics.models.DP.components.PointTransformerV3.model import PointTransformerV3, offset2bincount, offset2batch

# Policy
from zero.expForwardKinematics.models.FK.component.DA3D_layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule,
    ParallelAttention,
)
from zero.expForwardKinematics.models.FK.component.DA3D_position_encodings import (
    SinusoidalPosEmb,
    RotaryPositionEncoding3D,)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from codebase.z_model.attentionlayer import SelfAttnFFW, CrossAttnFFW, PositionalEncoding
from zero.expForwardKinematics.ReconLoss.FrankaPandaFK import FrankaEmikaPanda
from zero.expForwardKinematics.models.Base.BaseAll import BaseActionHead, BaseFeatureExtractor, BasePolicy
from codebase.z_model.positional_encoding import PositionalEncoding1D
# Trainer
from zero.expForwardKinematics.config.default import get_config
# Utils
'''
Policy
ActionHead, Feature Extractor
'''

# ---------------------------------------------------------------
# region 1. Feature Extractor


class FeatureExtractorPTv3Clean(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        self.config = config

        n_heads = config['FK']['ActionHead']['n_heads']
        d_ffw = config['FK']['ActionHead']['d_ffw']
        d_features = config['FK']['FeatureExtractor']['ptv3']['enc_channels'][-1]
        n_features = config['FK']['ActionHead']['n_features']

        self.ptv3_model = PointTransformerV3(**config['FK']['FeatureExtractor']['ptv3'])
        # self.cross_attn = CrossAttnFFW(d_features, n_heads, d_ffw, 0.1)

        self.n_features = n_features
        self.d_features = d_features

    def forward(self, batch):
        '''
        ptv3_batch: dict,dont care

        return [batch, feat_dim] .
        '''
        batch = self.prepare_ptv3_batch(batch)
        point = self.ptv3_model(batch)
        pcd_feat = torch.split(point['feat'], offset2chunk(point['offset']), dim=0)  # alarm
        pcd_feat, mask = pad_pcd_features(pcd_feat, self.n_features)
        return pcd_feat, mask

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


class ActionHeadToy(BaseActionHead):
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
            # SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
            # SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
            # SelfAttnFFW(d_model, n_heads, d_ffw, 0.1),
        ])

        self.CrossAttnList = nn.ModuleList([
            CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
            # CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
            # CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
            # CrossAttnFFW(d_model, n_heads, d_ffw, 0.1),
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


class ActionHeadDA3D(BaseActionHead):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # action head
        d_model = config['FK']['ActionHead']['d_model']
        d_instr = config['FK']['ActionHead']['d_instr']
        d_ffw = config['FK']['ActionHead']['d_ffw']
        n_heads = config['FK']['ActionHead']['n_heads']
        d_pcd_features = config['FK']['FeatureExtractor']['ptv3']['enc_channels'][-1]

        # encode
        self.action_hist_encoder = nn.Sequential(
            nn.Linear(8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.action_futr_noisy_encoder = nn.Sequential(
            nn.Linear(8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.time_embbedding = nn.Linear(1, d_model, bias=False)
        self.PE = PositionalEncoding1D(d_model)

        self.act_hist_query_obs_features = FFWRelativeSelfCrossAttentionModule(
            d_model, n_heads, 2, 1, use_adaln=False)

        self.act_futr_query_instr = FFWRelativeCrossAttentionModule(
            d_model, n_heads, 2, use_adaln=False)

        # main cross attn
        self.cross_attn = FFWRelativeCrossAttentionModule(
            d_model, n_heads, 2, use_adaln=True)

        self.self_attn = FFWRelativeSelfAttentionModule(
            d_model, n_heads, 4, use_adaln=True)

        # decoder
        self.JP_futr_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 8)
        )

        self.vl_cross_attn = FFWRelativeCrossAttentionModule(
            d_model, n_heads, 2, use_adaln=False)

    def forward(self,
                JP_futr_noisy, JP_hist,
                obs_features, obs_features_mask,
                instr, instr_mask,  # TODO: instr mask
                t):

        # encode features
        t = self.time_embbedding(t.unsqueeze(-1).float())
        act_hist = self.action_hist_encoder(JP_hist)
        act_futr_noisy = self.action_futr_noisy_encoder(JP_futr_noisy)

        # act_hist_query_obs_features， adaln_gripper_feature

        act_hist = self.act_hist_query_obs_features(q=act_hist + self.PE(act_hist), k=obs_features + self.PE(obs_features), key_padding_mask=obs_features_mask)[-1]

        # instruction
        act_futr_noisy = self.act_futr_query_instr(q=act_futr_noisy + self.PE(act_futr_noisy), k=instr + self.PE(instr), key_padding_mask=instr_mask)[-1]
        act_futr_noisy = act_futr_noisy + self.PE(act_futr_noisy)

        # 所谓predictionhead,对应3dda的DiffusionHead的prediction_head
        # t = t + act_hist  # 对应3dda的DiffusionHead的endoe_denoising_timestep

        act_futr_noisy = self.cross_attn(q=act_futr_noisy + self.PE(act_futr_noisy), k=obs_features + self.PE(obs_features), key_padding_mask=obs_features_mask, diff_ts=t)[-1]

        features = torch.cat([act_futr_noisy, obs_features], dim=1)
        features = features + self.PE(features)

        # self attn
        mask = torch.cat([torch.zeros(act_futr_noisy.shape[:2], device=act_futr_noisy.device), obs_features_mask], dim=1)
        features = self.self_attn(q=features, key_padding_mask=mask)[-1][:, :8, :]

        action_pred = self.JP_futr_decoder(features)
        return action_pred


class ActionHeadFKV1(BaseActionHead):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # action head
        d_model = config['FK']['ActionHead']['d_model']
        d_instr = config['FK']['ActionHead']['d_instr']
        d_ffw = config['FK']['ActionHead']['d_ffw']
        n_heads = config['FK']['ActionHead']['n_heads']
        d_pcd_features = config['FK']['FeatureExtractor']['ptv3']['enc_channels'][-1]
        n_cross_attn_layers = config['FK']['ActionHead']['n_cross_attn_layers']
        n_self_attn_layers = config['FK']['ActionHead']['n_self_attn_layers']
        # encode
        self.action_hist_encoder = nn.Sequential(
            nn.Linear(8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.action_futr_noisy_encoder = nn.Sequential(
            nn.Linear(8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.time_embbedding = nn.Linear(1, d_model, bias=False)
        self.PE = PositionalEncoding1D(d_model)

        # main cross attn
        self.cross_attn = FFWRelativeCrossAttentionModule(
            d_model, n_heads, n_cross_attn_layers, use_adaln=True)

        self.self_attn = FFWRelativeSelfAttentionModule(
            d_model, n_heads, n_self_attn_layers, use_adaln=True)

        # decoder
        self.JP_futr_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 8)
        )

        self.vl_cross_attn = FFWRelativeCrossAttentionModule(
            d_model, n_heads, 2, use_adaln=False)

    def forward(self,
                JP_futr_noisy, JP_hist,
                obs_features, obs_features_mask,
                instr, instr_mask,  # TODO: instr mask
                t):

        # 0.1 encode features
        t = self.time_embbedding(t.unsqueeze(-1).float())
        JP_hist = self.action_hist_encoder(JP_hist)
        JP_futr_noisy = self.action_futr_noisy_encoder(JP_futr_noisy)

        # 0.2 add positional encoding

        # 1. act_hist with obs_features
        # 2. cross attn on
        JP_hist_mask = torch.ones(JP_hist.shape[:2], device=JP_hist.device, dtype=torch.bool)

        features_obs_instr = torch.cat([obs_features, instr, JP_hist], dim=1)
        features_mask = torch.cat([obs_features_mask, instr_mask, JP_hist_mask], dim=1)

        JP_futr_noisy = JP_futr_noisy + self.PE(JP_futr_noisy)
        features_obs_instr = features_obs_instr + self.PE(features_obs_instr)
        JP_futr_noisy = self.cross_attn(q=JP_futr_noisy, k=features_obs_instr, key_padding_mask=~features_mask, diff_ts=t)[-1]

        # 3. act_futr_noisy with obs_features
        features = torch.cat([JP_futr_noisy, obs_features], dim=1)  # no need pe, 前面已经加过了
        inv_mask = torch.cat([torch.zeros(JP_futr_noisy.shape[:2], device=JP_futr_noisy.device), ~obs_features_mask], dim=1)
        features = self.self_attn(q=features, diff_ts=t, key_padding_mask=inv_mask)[-1][:, :8, :]

        action_pred = self.JP_futr_decoder(features)
        return action_pred


# endregion
# ---------------------------------------------------------------
# region 3. Policy
class PolicyFK(BasePolicy):
    def __init__(self, config):
        super().__init__()

        # Policy itself
        self.ActionHead = ActionHeadFKV1(config)
        self.FeatureExtractor = FeatureExtractorPTv3Clean(config)
        self.config = config
        self.franka = FrankaEmikaPanda()
        self.action_theta_offset = [0, 0, 0, math.radians(-4), 0, 0, 0]

        # DDPM
        self.scheduler = DDPMScheduler(
            beta_start=config['FK']['Policy']['DDPM']['beta_1'],
            beta_end=config['FK']['Policy']['DDPM']['beta_T'],
            beta_schedule="scaled_linear",
            num_train_timesteps=config['FK']['Policy']['DDPM']['T'],
            prediction_type="epsilon",
        )

        try:
            self.horizon = config['FK']['ActionHead']['horizon']
        except:
            self.horizon = 8

    def forward(self, batch):
        # get data
        JP_futr_0 = batch['JP_futr']  # [batch, horizon, action_dim]
        JP_hist = batch['JP_hist']  # [batch, horizon, action_dim]
        instr = batch['instr']
        instr_mask = batch['instr_mask']
        noncollision_mask = batch['noncollision_mask']  # [batch, horizon, action_dim]

        # feature extraction

        obs_features, obs_features_mask = self.FeatureExtractor(batch)

        noise = torch.randn(JP_futr_0.shape, device=JP_futr_0.device)
        # DDPM scheduler
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        JP_futr_t = self.scheduler.add_noise(JP_futr_0, noise, timesteps)

        noise_pred = self.ActionHead(JP_futr_t, JP_hist,
                                     obs_features, obs_features_mask,
                                     instr, instr_mask,  # TODO: instr mask
                                     t=timesteps)  # [batch, horizon, action_dim]

        loss = F.l1_loss(noise_pred, noise, reduction='mean')

        return loss

    @torch.no_grad()
    def inference_one_sample_JP(self, batch):
        '''
        actionhead input:(
                JP_futr_noisy, JP_hist,
                obs_features, obs_features_mask,
                instr, instr_mask,
                t)
        '''

        JP_futr_noisy_x_t = None
        JP_hist = batch['JP_hist']  # [batch, horizon, action_dim]
        obs_features, obs_features_mask = self.FeatureExtractor(batch)
        instr = batch['instr']
        instr_mask = batch['instr_mask']
        noncollision_mask = batch['noncollision_mask']  # TODO: inference 用不到，但用不到是不合理的,因此，放在这里，以便后用

        B = obs_features.shape[0]
        JP_futr_noisy_x_t = torch.randn(B, self.horizon, 8).to(obs_features.device)  # [B, horizon, action_dim]

        # inverse part
        timesteps = self.scheduler.timesteps
        for t in timesteps:
            noise_pred = self.ActionHead(JP_futr_noisy_x_t, JP_hist,
                                         obs_features, obs_features_mask,
                                         instr, instr_mask,  # TODO: instr mask
                                         t=t * torch.ones(len(obs_features)).to(obs_features.device).long()
                                         )
            JP_futr_noisy_x_t = self.scheduler.step(noise_pred, t, JP_futr_noisy_x_t).prev_sample

        x_0 = JP_futr_noisy_x_t
        return torch.clip(x_0, -1, 1)  # [B, horizon, action_dim]


class PolicyV111(BasePolicy):
    def __init__(self, config):
        super().__init__()

        # Policy itself
        self.ActionHead = ActionHeadFKV1(config)
        self.FeatureExtractor = FeatureExtractorPTv3Clean(config)
        self.config = config
        self.franka = FrankaEmikaPanda()
        self.action_theta_offset = [0, 0, 0, math.radians(-4), 0, 0, 0]

        # DDPM
        # adapted from
        # https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py

        beta_1 = config['FK']['Policy']['DDPM']['beta_1']
        beta_T = config['FK']['Policy']['DDPM']['beta_T']
        t = config['FK']['Policy']['DDPM']['T']

        def rb(name, val): return self.register_buffer(name, val)  # 这一步太天才了

        rb('betas', torch.linspace(beta_1, beta_T, t).double())

        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:t]

        rb('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        rb('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # denoising coeffs
        rb('coeff1', torch.sqrt(1. / alphas))
        rb('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.t_max = t

        #
        self.collision_loss_flag = config['FK']['Policy']['collision_loss']
        try:
            self.horizon = config['FK']['ActionHead']['horizon']
        except:
            self.horizon = 8

    def forward(self, batch):
        # get data
        JP_futr_0 = batch['JP_futr']  # [batch, horizon, action_dim]
        JP_hist = batch['JP_hist']  # [batch, horizon, action_dim]
        instr = batch['instr']
        instr_mask = batch['instr_mask']
        noncollision_mask = batch['noncollision_mask']  # [batch, horizon, action_dim]

        # feature extraction

        obs_features, obs_features_mask = self.FeatureExtractor(batch)

        # DDPM scheduler
        t = torch.randint(self.t_max, size=(JP_futr_0.shape[0], ), device=JP_futr_0.device)
        noise = torch.rand_like(JP_futr_0)
        JP_futr_t = self._q_sample(JP_futr_0, t, noise)  # [batch, horizon, action_dim]

        # DDPM predict
        noise_pred = self.ActionHead(JP_futr_t, JP_hist,
                                     obs_features, obs_features_mask,
                                     instr, instr_mask,  # TODO: instr mask
                                     t=t)  # [batch, horizon, action_dim]

        JP_futr_0_pred = self._inverse_q_sample(JP_futr_t, t, noise_pred)  # [batch, horizon, action_dim]

        # loss
        ddpm_loss = F.l1_loss(noise_pred, noise, reduction='mean')

        if self.collision_loss_flag:
            collision_loss = self._collision_loss(batch['pc_fts'][:, :3], batch['npoints_in_batch'], JP_futr_0_pred, noncollision_mask)
        else:
            collision_loss = 0

        loss = 30 * ddpm_loss + 10 * collision_loss
        return loss

    @torch.no_grad()
    def inference_one_sample_JP(self, batch):
        '''
        actionhead input:(
                JP_futr_noisy, JP_hist,
                obs_features, obs_features_mask,
                instr, instr_mask,
                t)

        '''
        JP_futr_noisy = None
        JP_hist = batch['JP_hist']  # [batch, horizon, action_dim]
        obs_features, obs_features_mask = self.FeatureExtractor(batch)
        instr = batch['instr']
        instr_mask = batch['instr_mask']
        noncollision_mask = batch['noncollision_mask']  # TODO: inference 用不到，但用不到是不合理的,因此，放在这里，以便后用

        B = obs_features.shape[0]
        x_t = torch.randn(B, self.horizon, 8).to(obs_features.device)  # [B, horizon, action_dim]

        for time_step in reversed(range(self.t_max)):
            print('time_step', time_step)
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            JP_futr_noisy = x_t
            actionhead_input = (JP_futr_noisy, JP_hist,
                                obs_features, obs_features_mask,
                                instr, instr_mask,
                                t)
            mean, var = self._p_mean_variance(x_t=x_t, t=t, actionhead_input=actionhead_input)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)  # [B, horizon, action_dim]

    # region 3.1 DDPM
    # Forward Part of Diffusion,
    def _q_sample(self, x_0, t, noise):
        '''
        forward diffusion process, 加噪声
        '''
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t

    def _inverse_q_sample(self, x_t, t, noise):
        '''
        inverse diffusion process, it is not denoising! Just for apply pysical rules
        '''
        sqrt_alphas_bar = extract(self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        x_0 = (x_t - sqrt_one_minus_alphas_bar * noise) / sqrt_alphas_bar
        return x_0

    # Backward Part of Diffusion,
    @torch.no_grad()
    def _p_mean_variance(self, x_t, t, actionhead_input):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.ActionHead(*actionhead_input)
        xt_prev_mean = self._predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    @torch.no_grad()
    def _predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    # region 3.2 Collision Loss

    def _collision_loss(self, xyz, n_xyz, JP, noncollision_mask):
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
        P_kb_minus_box = torch.abs(P_kp[..., :3]) - torch.abs(bbox)
        invasion = self._get_invasion(P_kb_minus_box, mask)  # TODO: need a more reasonable invasion

        collision_loss = torch.abs(invasion)
        return collision_loss

    def _get_invasion(self, P_kb_minus_box, mask: Tensor):
        '''
        暂时这样，
        理想状态下，invasion与n_negative正比，与p_kb_minus_box的数值成正比
        '''
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # for boardcasting
        idx = P_kb_minus_box < 0  # negative meas inside box
        test = torch.sum(idx)
        idx = idx & mask
        n_negative = torch.sum(idx)
        invasion = torch.sum(P_kb_minus_box[idx]) / n_negative
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
        mask[i, cur_len:] = False

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
    config_path = '/data/zero/zero/expForwardKinematics/config/FK.yaml'
    config = get_config(config_path)
    FeatureExtractorPTv3Clean(config)


# test()
