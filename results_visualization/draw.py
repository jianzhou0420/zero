import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open('/media/jian/ssd4t/zero/results_visualization.py/close_jar_0_1000/metrics.csv', 'r') as f:
    df = pd.read_csv(f)

print(df)


# 创建图形和第一个 y 轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制第一条折线 (y1)
ax1.plot(x1, y2, label='Loss', color='blue', linewidth=2)
ax1.set_xlabel('Epoches', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.5)

# 创建第二个 y 轴，共享 x 轴
ax2 = ax1.twinx()
ax2.plot(x1, y1, label='SR', color='red', linewidth=2)
ax2.set_ylabel('SR', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 添加图例
ax1.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

# 添加标题
plt.title('Close_Jar', fontsize=16)

# 显示图形
plt.show()
