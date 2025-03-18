from einops import reduce
from zero.expAugmentation.models.dp2d.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch
import torch.nn as nn
from zero.expAugmentation.models.dp2d.PointTransformerV3.model import (
    PointTransformerV3, offset2bincount, offset2batch
)
from zero.expAugmentation.models.lotus.PointTransformerV3.model_ca import PointTransformerV3CA
from zero.expAugmentation.config.default import build_args
from zero.expAugmentation.models.Base.BaseAll import BaseActionHead, BaseFeatureExtractor, BasePolicy
import numpy as np

# 先不要参数化
# 先不要大改，按照DP的写法来


'''
Policy
ActionHead, Feature Extractor

'''

# ---------------------------------------------------------------
# region 0. Some tools


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


class FeatureExtractor(BaseFeatureExtractor):
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
# endregion
# ---------------------------------------------------------------
# region 2. ActionHead


class ActionHeadDP1d(BaseActionHead):
    def __init__(self, config):
        super().__init__()

        # 0. in params
        horizon = int(config.ActionHead.horizon)
        action_dim = int(config.ActionHead.action_dim)
        global_cond_dim = config.ActionHead.global_cond_dim
        diffusion_step_embed_dim = config.ActionHead.diffusion_step_embed_dim

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100
        )
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        print('num_inference_steps:', self.num_inference_steps)
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=[256, 512, 1024],
            kernel_size=3,
            n_groups=8,
        )
        # 3. out params
        self.action_shape = (horizon, action_dim)

    # inference
    def inference_one_sample(self, cond):
        action_shape = self.action_shape
        # 原本的condition data 具有误导性，全删了，trajectory,不要condition

        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(action_shape).unsqueeze(0).to('cuda')

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            timestep = torch.tensor([t]).long().unsqueeze(0).to('cuda')
            # cond = cond.squeeze(0)
            model_output = model(trajectory, timestep, local_cond=None, global_cond=cond)

            trajectory = scheduler.step(model_output, t, trajectory,).prev_sample

        return trajectory

    # training
    def forward(self, actions, cond):
        '''
        action: [batch, horizon, action_dim]
        cond: [batch, feat_dim]
        '''
        bs = actions.size(0)
        # add noise to the action
        epsilon_noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,)).long().unsqueeze(1).to('cuda')

        noisy_actions = self.noise_scheduler.add_noise(actions, epsilon_noise, timesteps)
        # predict
        pred = self.model(noisy_actions, timesteps, local_cond=None, global_cond=cond)

        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = epsilon_noise
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = noisy_actions
        else:
            raise NotImplementedError
        loss = nn.functional.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
# endregion
# ---------------------------------------------------------------
# region 3. Policy


class PolicyPtv3DP1d(BasePolicy):
    def __init__(self, config):
        super().__init__()
        self.ActionHead = ActionHeadDP1d(config)
        self.FeatureExtractor = FeatureExtractor(config)

        self.config = config

    def forward(self, batch):
        ptv3_batch = self.FeatureExtractor.prepare_ptv3_batch(batch)
        features = self.FeatureExtractor(ptv3_batch)
        loss = self.ActionHead.forward(batch['theta_positions'], features)
        return loss

    def inference_one_sample(self, batch):
        ptv3_batch = self.FeatureExtractor.prepare_ptv3_batch(batch)
        cond = self.FeatureExtractor(ptv3_batch)
        return self.ActionHead.inference_one_sample(cond)

# endregion


def test():
    import pickle
    config_path = '/data/zero/zero/expAugmentation/config/DP.yaml'
    config = build_args(config_path)
    policy = PolicyPtv3DP1d(config)
    example_data = '/data/zero/1_Data/C_Dataset_Example/example.pkl'
    with open(example_data, 'rb') as f:
        data = pickle.load(f)
    lotus_batch = ptv3_collate_fn([data])

    for key in lotus_batch.keys():
        if isinstance(lotus_batch[key], torch.Tensor):
            lotus_batch[key] = lotus_batch[key].to('cuda')
        if isinstance(lotus_batch[key], list):
            try:
                if isinstance(lotus_batch[key][0], torch.Tensor):
                    lotus_batch[key] = [x.to('cuda') for x in lotus_batch[key]]
            except:
                pass

    print(lotus_batch.keys())
    print(policy.forward(lotus_batch))


# test()
