import torch


tensor1 = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
tensor2 = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
print(tensor1)

print(tensor1.reshape(5, 2))
print(tensor2.permute(1, 0))
