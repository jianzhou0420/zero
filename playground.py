import numpy as np

test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test = test.astype(np.float128)

print(test.dtype)
print(test.dtype == np.float_)
