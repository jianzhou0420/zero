import torch

# Original list of tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.tensor([7, 8, 9])

pred_actions = [a, b, c]

# Unpacking
pred_pos, pred_rot, pred_open = pred_actions

# Check memory identity
print(pred_pos is pred_actions[0])  # True
print(pred_rot is pred_actions[1])  # True
print(pred_open is pred_actions[2])  # True

# Modifying pred_pos affects pred_actions[0]
pred_pos[0] = 100
print(pred_actions[0])  # tensor([100, 2, 3])
