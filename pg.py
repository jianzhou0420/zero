import torch

# 假设的变换矩阵：形状 [8,7,4,4]
T = torch.randn(8, 7, 4, 4)

# 假设的点云：形状 [N,3]
N = 100  # 示例点数
points = torch.randn(N, 3)

# 1. 转换为齐次坐标：得到 [N,4]
ones = torch.ones(N, 1, device=points.device)
points_homo = torch.cat([points, ones], dim=1)

# 2. 转换为列向量形式，并扩展批次维度：
#    先转换为列向量：[N,4] -> [N,4,1]
points_homo = points_homo.unsqueeze(-1)  # 形状 [N,4,1]
#    再扩展维度以便广播：变为 [1,1,N,4,1]
points_homo = points_homo.unsqueeze(0).unsqueeze(0)

# 3. 应用变换
#    先扩展 T 的维度：T 原始形状 [8,7,4,4]，在第三个维度加一个 1 -> [8,7,1,4,4]
T_expanded = T.unsqueeze(2)
#    进行矩阵乘法：结果形状为 [8,7,N,4,1]
transformed = torch.matmul(T_expanded, points_homo)
#    squeeze 掉最后一个维度，得到 [8,7,N,4]
transformed = transformed.squeeze(-1)

# 4. 从齐次坐标转换为 3D 坐标
#    注意：通常第四个分量为 1，但若不为 1，需要除以第四个分量
transformed_points = transformed[..., :3] / transformed[..., 3:4]

# 最终 transformed_points 的形状为 [8,7,N,3]
print(transformed_points.shape)  # 输出：torch.Size([8, 7, N, 3])
