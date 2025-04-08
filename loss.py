import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 数据
data = pd.read_csv("/data/zero/2_Train/DP/Mar15_17-38-06_DP/version_0/metrics.csv")

# 假设你要用 train_loss_epoch 作为每个 epoch 的 loss，
# 这里计算每个 epoch 的平均 train_loss_epoch（如果每个 epoch 有多条记录）
epoch_loss = data.groupby("epoch")["train_loss"].mean()

plt.figure(figsize=(8, 6))
plt.plot(epoch_loss.index, epoch_loss.values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss (Epoch)')
plt.title('Train Loss vs Epoch')
plt.grid(True)
plt.show()
