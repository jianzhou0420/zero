
'''
this script is to process the raw observation data to static data
to use it just 

'''
import argparse
import re
import os
from zero.expForwardKinematics.trainer_FK_all import OBS_FACTORY, CONFIG_FACTORY
from zero.expForwardKinematics.ObsProcessor.ObsProcessorFKAll import ObsProcessorRLBenchBase
from zero.expForwardKinematics.config.default import get_config
import pickle
import yacs.config
from tqdm import tqdm
from zero.z_utils.coding import check_and_make, natural_sort_key
from termcolor import cprint
import copy


class StaticProcess:  # just for code organization

    @staticmethod
    def run(config: yacs.config.CfgNode,
            ObsProcessor: ObsProcessorRLBenchBase,
            obs_raw_path: str,
            save_root: str):
        '''
        TODO
        '''
        cprint("=" * 20, 'blue')
        cprint(type(ObsProcessor), 'blue')
        cprint(f"obs_raw_path:{obs_raw_path} ", 'blue')
        cprint(f"save_root:{save_root} ", 'blue')
        cprint("=" * 20, 'blue')

        episodes_list = StaticProcess.get_all_episodes(obs_raw_path)
        pbar = tqdm(total=len(episodes_list))
        for i, s_dict in tqdm(enumerate(episodes_list)):
            relative_path = os.path.join(str(s_dict['task']), str(s_dict['variation']), str(s_dict['episode']))
            abs_path = os.path.join(save_root, relative_path)

            with open(os.path.join(s_dict['path'], 'data.pkl'), 'rb') as f:
                obs_raw = pickle.load(f)
            static_data = ObsProcessor.static_process(obs_raw)

            save_path = os.path.join(abs_path, 'data.pkl')
            check_and_make(os.path.dirname(save_path))
            with open(os.path.join(save_path), 'wb') as f:
                pickle.dump(static_data, f)
            pbar.update(1)

    @staticmethod
    def get_all_episodes(obs_raw_path):
        tasks_list = sorted(os.listdir(obs_raw_path), key=natural_sort_key)
        episodes_dict_list = []

        for i, task in enumerate(tasks_list):
            this_task_path = os.path.join(obs_raw_path, task)
            variations_list = sorted(os.listdir(this_task_path), key=natural_sort_key)
            for j, variation in enumerate(variations_list):
                single_episode_dict = {
                    'task': None,
                    'variation': None,
                    'episode': None,
                    'path': None,
                }
                this_variation_path = os.path.join(this_task_path, variation, 'episodes')
                episodes_list = sorted(os.listdir(this_variation_path), key=natural_sort_key)
                for k, episode in enumerate(episodes_list):
                    this_episode_path = os.path.join(this_variation_path, episode)
                    single_episode_dict['task'] = copy.copy(task)
                    single_episode_dict['variation'] = copy.copy(variation)
                    single_episode_dict['episode'] = copy.copy(episode)
                    single_episode_dict['path'] = copy.copy(this_episode_path)
                    episodes_dict_list.append(copy.deepcopy(single_episode_dict))

        return episodes_dict_list


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Train FK')
    argparser.add_argument('--model', type=str, default='DP_traj')
    argparser.add_argument('--obs_raw_path', type=str, default='./1_Data/A_Selfgen/trajectory/test2/42')
    args = argparser.parse_args()

    config = get_config(CONFIG_FACTORY[args.model])
    ObsProcessor = OBS_FACTORY[args.model](config, train_flag=True)

    obs_raw_path = args.obs_raw_path
    tail_path = obs_raw_path.split('A_Selfgen')[1][1:]
    save_root = os.path.join("./1_Data/B_Preprocess", f"{args.model}", tail_path)
    StaticProcess.run(config, ObsProcessor, obs_raw_path, save_root)
