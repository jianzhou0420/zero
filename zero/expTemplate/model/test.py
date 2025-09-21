import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Any
from torch import Tensor
from abc import abstractmethod

# ----------------------------------------------------------------------------
# 您提供的 BaseVAE (无需修改)
# ----------------------------------------------------------------------------


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


# ----------------------------------------------------------------------------
# 新的 SequentialVAE 模型
# 这个模型可以处理 (B, 16, 7) -> (B, 16, 10) 的转换
# ----------------------------------------------------------------------------
class SequentialVAE(BaseVAE):
    """
    一个使用全连接层处理序列数据的VAE模型。
    它将输入数据从 (B, seq_len, in_features) 编码到潜在空间，
    然后解码为 (B, seq_len, out_features) 的形状。
    """

    def __init__(self,
                 sequence_len: int,
                 input_features: int,
                 output_features: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None,
                 **kwargs) -> None:
        super(SequentialVAE, self).__init__()

        self.latent_dim = latent_dim
        self.sequence_len = sequence_len
        self.input_features = input_features
        self.output_features = output_features

        # 计算扁平化后的输入和输出维度
        input_dim = sequence_len * input_features   # 16 * 7 = 112
        output_dim = sequence_len * output_features  # 16 * 10 = 160

        if hidden_dims is None:
            # 您可以根据需要调整这些隐藏层的维度
            hidden_dims = [256, 128]

        # ---- 构建编码器 ----
        encoder_modules = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_modules)

        # 编码器输出到均值和方差
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # ---- 构建解码器 ----
        decoder_modules = []

        # 解码器从潜在空间开始
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        # 反转隐藏层维度用于解码
        hidden_dims.reverse()

        in_dim = hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            in_dim = hidden_dims[i + 1]

        # 最后一个解码层映射到最终的输出维度
        decoder_modules.append(
            nn.Sequential(
                nn.Linear(in_dim, output_dim),
                nn.Tanh()  # 使用Tanh将输出值缩放到[-1, 1]范围，可根据需要更换
            )
        )
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        编码输入张量。
        :param input: (Tensor) 输入张量 [B, 16, 7]
        :return: (List[Tensor]) 包含均值和对数方差的列表
        """
        # 将输入扁平化以适应全连接层
        result = torch.flatten(input, start_dim=1)  # [B, 16 * 7]
        result = self.encoder(result)

        # 计算均值和对数方差
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        将潜在空间向量z映射回目标数据空间。
        :param z: (Tensor) 潜在向量 [B, latent_dim]
        :return: (Tensor) 重建的输出张量 [B, 16, 10]
        """
        result = self.decoder(z)  # [B, 16 * 10]
        # 将扁平的输出重塑为序列形状
        result = result.view(-1, self.sequence_len, self.output_features)  # [B, 16, 10]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        重参数化技巧，用于从 N(mu, var) 进行采样。
        :param mu: (Tensor) 均值 [B, D]
        :param logvar: (Tensor) 对数方差 [B, D]
        :return: (Tensor) 采样的潜在向量 [B, D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Dict[str, Tensor]:
        """
        模型的前向传播。
        :param input: (Tensor) 输入数据 [B, 16, 7]
        :return: (Dict) 包含重建输出、输入、均值和对数方差的字典
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)

        # 返回一个字典，使其更清晰
        return {
            "reconstruction": reconstruction,  # [B, 16, 10]
            "input": input,                   # [B, 16, 7]
            "mu": mu,
            "log_var": log_var
        }

    def loss_function(self,
                      prediction: Tensor,
                      target: Tensor,
                      mu: Tensor,
                      log_var: Tensor,
                      kld_weight: float) -> Dict[str, Tensor]:
        """
        计算VAE的损失函数。
        注意：这里的重建损失是计算`prediction`和`target`之间的差距。
        在训练循环中，你需要同时提供模型的输入和期望的目标输出。

        :param prediction: (Tensor) VAE解码器的输出 [B, 16, 10]
        :param target: (Tensor) 期望的目标输出 [B, 16, 10]
        :param mu: (Tensor) 潜在均值
        :param log_var: (Tensor) 潜在对数方差
        :param kld_weight: (float) KLD损失的权重
        :return: (Dict) 包含总损失和各个部分损失的字典
        """
        # 重建损失
        recons_loss = F.mse_loss(prediction, target)

        # KLD损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # 总损失
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    # --- 以下方法继承自BaseVAE但在此处未实现，您可以根据需要实现 ---
    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)["reconstruction"]


# ----------------------------------------------------------------------------
# 如何使用 SequentialVAE
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # ---- 1. 定义模型参数 ----
    batch_size = 32
    seq_len = 16
    input_features = 7
    output_features = 10
    latent_dim = 20  # 潜在空间维度，可以调整
    hidden_dims = [256, 128]  # 隐藏层维度，可以调整

    # ---- 2. 实例化模型 ----
    model = SequentialVAE(
        sequence_len=seq_len,
        input_features=input_features,
        output_features=output_features,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )

    print("模型结构:")
    print(model)

    # ---- 3. 创建模拟数据 ----
    # 模拟模型的输入数据
    input_data = torch.randn(batch_size, seq_len, input_features)
    # 模拟期望的目标输出数据 (用于计算损失)
    target_data = torch.randn(batch_size, seq_len, output_features)

    # ---- 4. 运行模型并计算损失 ----
    # 前向传播
    outputs = model(input_data)

    # 提取输出
    prediction = outputs["reconstruction"]
    mu = outputs["mu"]
    log_var = outputs["log_var"]

    print(f"\n输入形状: {input_data.shape}")
    print(f"输出形状: {prediction.shape}")

    # 检查形状是否符合预期
    assert prediction.shape == (batch_size, seq_len, output_features)

    # 计算损失
    # 注意：我们将`prediction`和`target_data`传入损失函数
    loss_dict = model.loss_function(
        prediction=prediction,
        target=target_data,  # 使用目标数据计算重建损失
        mu=mu,
        log_var=log_var,
        kld_weight=1.0  # KLD权重，通常在训练中会调整
    )

    print(f"\n计算得到的损失: {loss_dict}")
