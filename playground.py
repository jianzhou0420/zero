import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        # Cosine decay
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    def get_scheduler(self):
        return LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)


test = torch.nn.Sequential(
    nn.Linear(10, 10),
)
schedule = WarmupCosineSchedule(torch.optim.Adam(test.parameters(), lr=0.5), 100, 1000).get_scheduler()

for i in range(1000):
    lr = schedule.get_last_lr()
    print(f"Step {i}: lr={lr}")
