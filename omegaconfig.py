from omegaconf import OmegaConf

base_cfg = OmegaConf.load('/media/jian/ssd4t/zero/zero/expForwardKinematics/config/FK.yaml')
print(type(base_cfg))
