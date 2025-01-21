import argparse
import os
import yacs.config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
config_name = 'after_shock.yaml'

args.config = os.path.join('/workspace/zero/zero/v3/config/', config_name)

config = yacs.config.CfgNode(new_allowed=True)
config.merge_from_file(args.config)

print(config.TRAIN_DATASET.data_dir)
