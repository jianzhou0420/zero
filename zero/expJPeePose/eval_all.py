
import os
from zero.expForwardKinematics.config.default import get_config
from zero.z_utils.coding import natural_sort_key
from zero.expJPeePose.trainer_all import Trainer_all
import pickle
from tqdm import tqdm
from zero.z_utils.normalizer_action import denormalize_JP, denormalize_pos, ortho6d2quat, normalize_JP
from zero.z_utils.action_visualizer import visualizor
import numpy as np

import yaml
import torch


class Evaluator:
    '''
    just for code organization
    '''

    def __init__(self):
        eval_config = get_config('./zero/expJPeePose/config/eval.yaml')
        exp_dir = eval_config['exp_dir']
        ckpt_path_all = sorted(os.listdir(os.path.join(exp_dir, 'checkpoints')), key=natural_sort_key)
        if eval_config['epoch'] is not None:
            for i, ckpt_path in enumerate(ckpt_path_all):
                if f"epoch={eval_config['epoch']}" in ckpt_path:
                    ckpt_path = os.path.join(exp_dir, 'checkpoints', ckpt_path)
                    break
        else:
            ckpt_path = os.path.join(exp_dir, 'checkpoints', ckpt_path_all[-1])

        model_config_path = os.path.join(exp_dir, 'hparams.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.UnsafeLoader)['config']
        self.model = Trainer_all.load_from_checkpoint(ckpt_path, config=model_config)

    def inference_JP2eePose(self):
        test_data_path = "./1_Data/B_Preprocess/eePoseJP/data.pkl"
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            # eePose = data[0]
            eePose_hat = np.zeros(7)
            JP = data[1]
            JP = normalize_JP(JP)
            batch = {
                'input': torch.from_numpy(JP[:-1]).float().to(self.model.device),
                # 'output': eePose
            }
            PosOrtho6D_hat = self.model.policy.inference_one_sample(batch)

            PosOrtho6D_hat = PosOrtho6D_hat.detach().cpu().numpy()
            eePose_hat[:3] = denormalize_pos(PosOrtho6D_hat[:3])
            eePose_hat[3:] = ortho6d2quat(PosOrtho6D_hat[3:][None, None, :]).squeeze()
            JP = denormalize_JP(JP)
            visualizor.visualize_eePose_JP(eePose_hat, JP, return_o3d=False)
            # break

    def inference_eePose2JP(self):
        test_data_path = "./1_Data/B_Preprocess/eePoseJP/data.pkl"
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            eePose = data[0]
            JP = data[1]
            JP = normalize_JP(JP)
            batch = {
                'input': torch.from_numpy(eePose).float().to(self.model.device),
                # 'output': JP
            }
            JP_hat = self.model.policy.inference_one_sample(batch)
            JP_hat = JP_hat.detach().cpu().numpy()
            visualizor.visualize_eePose_JP(eePose, JP_hat, return_o3d=False)
            # break


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.inference_JP2eePose()
