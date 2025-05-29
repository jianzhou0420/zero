from typing import List
from zero.expForwardKinematics.trainer_FK_all import Trainer_all, OBS_FACTORY
from zero.expForwardKinematics.config.default import get_config
import yacs.config
from zero.expForwardKinematics.ObsProcessor import ObsProcessorRLBenchBase
from zero.z_utils.video_recorder import VideoRecorder
from zero.env.pusht.pusht_env import PushTEnv
from zero.env.pusht.pusht_image_env import PushTImageEnv


class EnvPushT:
    '''
    Accept eval_config, evaluate the model
    '''

    def __init__(self, eval_config: yacs.config.CfgNode):
        self.env = PushTImageEnv(
            legacy=False,
            block_cog=(256, 300),
            damping=0.9,
            render_size=128,
        )
        # obs_recorder buffer
        self.obs_list = []
        self.num_obs_hist = 2
        self.record_video = eval_config['record_video']

        # results path

        # record video
        if self.record_video:
            self.video_recorder = VideoRecorder(
                output_path=eval_config['video_output_path'],
                fps=eval_config['video_fps']
            )

    def eval_single_episode(self, eval_config: yacs.config.CfgNode):
        actioner = ActionerPushT(
            model_name=eval_config['config']['Trainer']['model_name'],
            checkpoint_path=eval_config['checkpoint'],
            config=eval_config['config']
        )

        obs = self.env.reset()
        self._update_obs_recorder(obs)
        done = False
        while not done:
            action = actioner.pred_action(self.obs_list)
            obs, reward, done, info = self.env.step(action)  # PushTImageEnv step 已经把obs处理好了
            self._update_obs_recorder(obs)
            if self.record_video:
                self.video_recorder.record_frame(obs['obs']['image'])

        self.env.close()

    def _update_obs_recorder(self, obs):
        if len(self.obs_list) == self.num_obs_hist:
            self.obs_list.pop(0)
            self.obs_list.append(obs)
        elif len(self.obs_list) == 0:
            [self.obs_list.append(obs) for _ in range(self.num_obs_hist)]
        elif len(self.obs_list) < self.num_obs_hist:
            num_to_add = self.num_obs_hist - len(self.obs_list)
            for i in range(num_to_add):
                self.obs_list.append(obs)
        else:
            raise ValueError(f"obs_recorder length is {len(self.obs_list)}, but it should be 0 or {self.num_obs_hist}.")


class ActionerPushT:
    '''
    Accept list of obs and return action
    '''

    def __init__(self, model_name: str, checkpoint_path: str, config: yacs.config.CfgNode):
        pl_trainer = Trainer_all.load_from_checkpoint(checkpoint_path, config)
        model = pl_trainer.policy
        obs_processor = OBS_FACTORY[model_name](config, train_flag=False)

        self.config = config
        self.model = model
        self.obs_processor = obs_processor

    def pred_action(self, obs_list):

        batch = self.obs_processor.obs2batch(obs_list)
        action = self.model.inference_one_sample(batch)
        return action


def main():
    pass
