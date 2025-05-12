import torch
import numpy as np

test = torch.zeros(1, 3, 224, 224)
test2 = np.zeros((1, 3, 224, 224))
a = test[:, 100:111, :, :]
b = test2[:, 100:111, :, :]
print(a.shape)
print(b.shape)
print(a)
