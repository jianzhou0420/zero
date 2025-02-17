import torch

coord = torch.rand(25565, 3)

# print(coord.shape)
print(coord.min(0))
