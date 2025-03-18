
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件，假设文件名为 'loss.csv'
df = pd.read_csv('/home/jian/Downloads/1.csv')

plt.figure(figsize=(8, 6))
plt.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss vs Epoch')
plt.ylim(0, 1)  # 设置纵坐标范围为 [0, 1]
plt.grid(True)
plt.show()
