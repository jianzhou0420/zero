import matplotlib.pyplot as plt
from zero.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
import time
from zero.common.replay_buffer import ReplayBuffer
from zero.v1.trainer_zero_test import TrainerTesterJazz
import yaml
import torch


class PushTAgent:
    def __init__(self):
        with open('/data/zero/zero/v1/config/zero_test.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.model = TrainerTesterJazz.load_from_checkpoint('/media/jian/data/ckpt/lightning_logs/version_2/checkpoints/last.ckpt', config_first_layer=config)

    def act(self, image):
        image = image / 255
        image = torch.tensor(image, dtype=torch.float32).cuda()
        data_dict = {'image': image}
        model_output = self.model._forward_pass_pusht(data_dict=data_dict)
        return tuple([int(model_output[0][0]), int(model_output[0][1])])


output = '.'
render_size = 96
# create PushT env with keypoints
replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
env = PushTKeypointsEnv(render_size=render_size, render_action=False, **kp_kwargs)
max_time_steps = 180
env.seed(42)
obs = env.reset()
agent = PushTAgent()
for i in range(180):
    # action = agent.act(obs)
    image = env.render('human').reshape(1, 3, 96, 96)
    action = agent.act(image)
    obs, reward, done, info = env.step(action)
    if done:
        print('done')
        break
    env.render('human')
    time.sleep(0.1)
