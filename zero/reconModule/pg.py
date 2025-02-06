import pickle
import numpy as np
from zero.expBins.models.lotus.utils.robot_box import RobotBox


new_data_path = '/data/zero/1_Data/A_Selfgen/train/20250204_225613/insert_onto_square_peg_peract/variation0/episodes/episode0/data.pkl'
with open(new_data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['sem'].shape)
print(data['sem'][0][0].shape)
