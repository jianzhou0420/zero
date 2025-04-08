import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

static_process = '/data/zero/1_Data/B_Preprocess/DA3D/close_jar/variation0/episodes/episode0/data.pkl'

with open(static_process, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

batch = data['rgb'][0]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    ax.imshow(batch[idx])
    ax.axis('off')  # Hide axes ticks
    ax.set_title(f"Image {idx+1}")

plt.tight_layout()
plt.show()
