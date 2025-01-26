import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
with open('/media/jian/ssd4t/zero/results_visualization.py/insert_peg_0.005/results.csv', 'r') as f:
    df = pd.read_csv(f)

epoch_number = []
sr = []
for index, row in df.iterrows():
    epoch_number.append(row['ckpt_name'].split('=')[-1].split('.')[0])
    sr.append(float(row['sr'].split('[')[-1].split(']')[0]))


with open('/media/jian/ssd4t/zero/results_visualization.py/close_jar_0_1000/metrics.csv', 'r') as f:
    df = pd.read_csv(f)

print(df)

loss = []
for index, row in df.iterrows():
    epoch = int(row['epoch'])
    try:
        loss[epoch] = float(row['train_loss_epoch'])
    except:
        loss.append(float(row['train_loss_epoch']))

sr.insert(0, 0)
epoch_number.insert(0, 0)
print(len(epoch_number))

print(len(sr))
print(len(loss))


x1 = np.arange(0, 1000, 1)

y1 = np.full_like(x1, np.nan, dtype=np.float16)
y2 = np.full_like(x1, np.nan, dtype=np.float16)

sx1 = epoch_number
sy1 = sr

loss = loss[:1000]


# for i, sx in enumerate(sx1):
#     y1[int(sx)] = sy1[i]
for i, sx in enumerate(sx1):
    if i != len(sx1) - 1:
        start_epoch = int(sx)
        end_epoch = int(sx1[i + 1])
        values = np.linspace(sy1[i], sy1[i + 1], end_epoch - start_epoch)
        y1[start_epoch:end_epoch] = values
    else:
        y1[int(sx)] = sy1[i]


for i in range(len(loss)):
    y2[i] = loss[i]


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
