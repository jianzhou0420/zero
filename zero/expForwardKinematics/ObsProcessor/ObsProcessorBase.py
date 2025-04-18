import torch
import torch.nn as nn


class ObsProcessorBase:
    def __init__(self, config):
        self.config = config

    def static_process_DA3D(self, x):
        return x

    def dynamic_process(self, x):
        return x
