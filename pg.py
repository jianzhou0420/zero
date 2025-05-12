import torch.nn as nn
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.nn.functional as F
# 1) Initialize the scheduler with the same training betas and timesteps
scheduler = DDPMScheduler(
    num_train_timesteps=100,    # match your training config
    beta_start=0.0001,
    beta_end=0.02,
)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class X0LossPlugin(nn.Module):
    def __init__(self, scheduler: DDPMScheduler):
        super().__init__()
        self.scheduler = scheduler

        def rb(name, val): return self.register_buffer(name, val)  # 这一步太天才了

        max_t = len(scheduler.timesteps)
        betas = scheduler.betas
        alphas = scheduler.alphas
        alphas_bar = scheduler.alphas_cumprod
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:max_t]

        rb('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        rb('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # denoising coeffs
        rb('coeff1', torch.sqrt(1. / alphas))
        rb('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

    def _inverse_q_sample(self, x_t, t, noise):
        '''
        inverse diffusion process, it is not denoising! Just for apply pysical rules
        '''
        sqrt_alphas_bar = extract(self.sqrt_alphas_bar, t, x_t.shape)
        sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        x_0 = (x_t - sqrt_one_minus_alphas_bar * noise) / sqrt_alphas_bar
        return x_0


test = X0LossPlugin(scheduler=scheduler)


x_t = torch.randn(2, 3, 4, 5)
t = torch.randint(0, 100, (2,))
noise = torch.randn_like(x_t)
x_0 = test._inverse_q_sample(x_t, t, noise)
print(x_0.shape)
