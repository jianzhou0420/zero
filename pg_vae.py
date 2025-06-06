import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=8, output_dim=10, hidden_dims=None, z_dim=4):
        """
        Args:
            input_dim  (int): Encoder 输入维度（这里是 8）。
            output_dim (int): Decoder 输出维度（这里是 10）。
            hidden_dims (list[int] or None): 中间隐藏层维度列表。如果为 None，会使用 [128, 64]。
            z_dim      (int): 潜在空间维度（latent dimension）。
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # ----------------------------
        # 1. Encoder：input_dim -> ... -> (mu, logvar)
        # ----------------------------
        enc_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h_dim))
            enc_layers.append(nn.BatchNorm1d(h_dim))
            enc_layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        # 到这里，prev_dim == hidden_dims[-1]
        self.encoder_net = nn.Sequential(*enc_layers)
        # 最终输出两个向量：mu 和 logvar，都维度为 z_dim
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)

        # ----------------------------
        # 2. Decoder：z_dim -> ... -> output_dim
        # ----------------------------
        dec_layers = []
        prev_dim = z_dim
        # 反向使用 hidden_dims 列表
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h_dim))
            dec_layers.append(nn.BatchNorm1d(h_dim))
            dec_layers.append(nn.ReLU(inplace=True))
            prev_dim = h_dim
        # 最后接一层 Linear 将维度映射到 output_dim
        dec_layers.append(nn.Linear(prev_dim, output_dim))
        # 如果你的输出在 [0,1] 之间，可以在外部 loss 中接 Sigmoid，这里不在模块内做
        self.decoder_net = nn.Sequential(*dec_layers)

    def encode(self, x):
        """
        输入 x 形状 (batch_size, input_dim)，输出 (mu, logvar)，形状都为 (batch_size, z_dim)。
        """
        h = self.encoder_net(x)        # (batch_size, hidden_dims[-1])
        mu = self.fc_mu(h)         # (batch_size, z_dim)
        logvar = self.fc_logvar(h)     # (batch_size, z_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick，从 N(mu, σ^2) 中采样。
        Args:
            mu:    (batch, z_dim)
            logvar:(batch, z_dim)
        Returns:
            z:     (batch, z_dim)
        """
        std = torch.exp(0.5 * logvar)    # sigma = exp(0.5 * logvar)
        eps = torch.randn_like(std)      # 从标准正态采样 (batch, z_dim)
        return mu + eps * std            # 结合均值和标准差

    def decode(self, z):
        """
        将潜在向量 z 解码为输出，维度为 (batch_size, output_dim)。
        """
        x_recon = self.decoder_net(z)
        return x_recon

    def forward(self, x):
        """
        整体前向：
            1. encode -> mu, logvar 
            2. reparameterize -> z 
            3. decode -> x_recon
        返回：
            x_recon: (batch, output_dim)
            mu, logvar 用于后续计算 KL 散度
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x_recon, x_target, mu, logvar, beta=1.0):
    """
    计算 VAE 的总损失：重建损失 + β * KL 散度
    Args:
        x_recon:    (batch, output_dim)   -> 重建的输出
        x_target:   (batch, output_dim)   -> 真实目标（这里如果输入与输出维度不同，x_target 指向输出；否则同输入）
        mu:         (batch, z_dim)
        logvar:     (batch, z_dim)
        beta:       KL项权重，LDM 的论文通常取 β=1
    Returns:
        loss:       标量
        recon_loss: 重建损失（MSE）
        kld_loss:   KL 散度损失
    """
    # 1) Reconstruction loss (MSE 形式)
    recon_loss = F.mse_loss(x_recon, x_target, reduction='mean')

    # 2) KL divergence: 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    #    先对每个维度算，最后按 batch 平均
    kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # 公式等价于: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_loss = torch.sum(kld_element).mul_(-0.5) / x_recon.size(0)

    loss = recon_loss + beta * kld_loss
    return loss, recon_loss, kld_loss


# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    # 假设 batch_size=16
    batch_size = 16
    input_dim = 8
    output_dim = 10
    z_dim = 4

    model = VAE(input_dim=input_dim, output_dim=output_dim, z_dim=z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 假设这里的 x_input 是一个 (16, 8) 的张量，
    # x_target 是 (16, 10) 的“真实”标签（可与 x_input 不同）。
    # 如果你要做自编码（input==output），可以令 x_target = x_input 并将 output_dim 设为 8。
    x_input = torch.randn(batch_size, input_dim)
    x_target = torch.randn(batch_size, output_dim)

    # 1. 前向
    x_recon, mu, logvar = model(x_input)

    # 2. 计算损失
    loss, recon_loss, kld_loss = vae_loss(x_recon, x_target, mu, logvar, beta=1.0)
    print(f"Total Loss: {loss.item():.4f}, "
          f"Reconstruction Loss: {recon_loss.item():.4f}, "
          f"KL Divergence Loss: {kld_loss.item():.4f}")
