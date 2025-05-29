import matplotlib.pyplot as plt
import numpy as np
from zero.env.pusht.pymunk_override import DrawOptions  # if needed for custom render
from zero.env.pusht.pusht_image_env import PushTEnv  # adjust import to where PushTEnv lives

# # 1. Instantiate the env (you can override defaults if you like)
# env = PushTEnv(
#     legacy=False,            # whether to use legacy set-state behavior
#     block_cog=(256, 300),    # optional: override block center-of-gravity
#     damping=0.9,             # optional: space.damping
#     render_action=True,      # draw the last action as a marker
#     render_size=128,         # size of returned RGB images
#     reset_to_state=None      # or supply a 5-vector to reset to a fixed state
# )
# # 2. Reset to start a new episode
# obs = env.reset()
# print("Initial observation:", obs)   # [agent_x, agent_y, block_x, block_y, block_angle]

# # 3. Step through with random actions, render and collect data
# frames = []
# done = False
# while not done:
#     # Sample a random valid action in the 2D workspace
#     action = env.action_space.sample()

#     # Advance the sim
#     obs, reward, done, info = env.step(action)

#     # Render to RGB array
#     frame = env.render(mode="rgb_array")
#     frames.append(frame)
#     plt.imshow(frame)
#     plt.show()
#     # Optional: print out some info
#     print(f"Obs={obs}, reward={reward:.3f}, done={done}")

# # 4. Close the window if you used human rendering
# env.close()

# # 5. (Optional) Save frames to disk or encode into a video using OpenCV, imageio, etc.
from zero.expForwardKinematics.models.Base.BaseAll import BasePolicy
import yacs.config
from zero.expForwardKinematics.trainer_FK_all import Trainer_all, OBS_FACTORY
from zero.expForwardKinematics.ObsProcessor import ObsProcessorRLBenchBase


class PushTActioner:
    def __init__(self,):
        pass

    def action(self,):
        pass


class PushTOnlineEvaluator:
    def __init__(self, eval_config: yacs.config.CfgNode):
        self.frames = []
        env = PushTEnv(
            legacy=False,
            block_cog=(256, 300),
            damping=0.9,
            render_action=True,
            render_size=128,
            reset_to_state=None
        )

        self.config = eval_config['config']
        model_name = self.config['Trainer']['model_name']
        pl_trainer = Trainer_all.load_from_checkpoint(eval_config['checkpoint'], config=self.config)
        model = pl_trainer.policy

        self.env = env
        self.model = model
        self.obs_processor = OBS_FACTORY[model_name](self.config, train_flag=False)  # type: ObsProcessorRLBenchBase
        self.obs_processor.dataset_init()

    def eval_single_episode(self,):
        obs = self.env.reset()
        done = False
        while not done:
            # Sample a random valid action in the 2D workspace
            action = self.model.inference_one_sample(obs)

            # Advance the sim
            obs, reward, done, info = self.env.step(action)

            # Render to RGB array
            frame = self.env.render(mode="rgb_array")
            self.frames.append(frame)
            plt.imshow(frame)
            plt.show()

    def process_obs(self, obs):
        # obs = obs['image']
        # obs = np.moveaxis(obs, 0, -1)
        return obs


class PushTDataCollector:
    def __init__(self,):
        pass

    def collect_data(self,):
        pass
