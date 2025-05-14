
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件，假设文件名为 'loss.csv'
df = pd.read_csv('/data/zero/2_Train/DP_traj_x0loss/0513_02/version_0/metrics.csv')

plt.figure(figsize=(8, 6))
plt.plot(df['epoch'], df['train_loss_epoch'], marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss vs Epoch')
# plt.ylim(0,)  # 设置纵坐标范围为 [0, 1]
plt.grid(True)
plt.savefig('train_loss.png')
