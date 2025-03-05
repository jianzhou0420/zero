import numpy as np


a = np.array([1, 0, 0])
arr = np.repeat(a[None, :], 100, axis=0)

print(arr.shape)
