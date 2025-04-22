from omegaconf import OmegaConf

base_cfg = OmegaConf.load('/data/zero/zero/zero/expForwardKinematics/config/FK.yaml')
print(type(base_cfg))
