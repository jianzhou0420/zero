import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

example_joint_postion_path = '/datacode/zero/1_Data/A_Selfgen/with_path_with_position/train/close_jar_peract/variation0/episodes/episode0/positions_all.pkl'
example_joint_postion_path = './1_Data/A_Selfgen/with_path_with_position/train/close_jar_peract/variation0/episodes/episode0/positions_all.pkl'

with open(example_joint_postion_path, 'rb') as f:
    data = pickle.load(f)


JOINT_POSITIONS_LIMITS = np.array([[-2.8973, 2.8973],
                                   [-1.7628, 1.7628],
                                   [-2.8973, 2.8973],
                                   [-3.0718, -0.0698],
                                   [-2.8973, 2.8973],
                                   [-0.0175, 3.7525],
                                   [-2.8973, 2.8973]])

# 'https://frankaemika.github.io/docs/control_parameters.html' 不确定RLBench的Coppeliasim是否遵循这个限制 TODO：verify it


def abs_position_2_relative_position(abs_action):
    '''
    abs_action: [7,] or [B, 7]
    JOINT_POSITIONS_LIMITS: [7, 2]
    '''
    if len(abs_action.shape) == 1:
        relative_action = abs_action - JOINT_POSITIONS_LIMITS[:, 0]
        relative_action = relative_action / (JOINT_POSITIONS_LIMITS[:, 1] - JOINT_POSITIONS_LIMITS[:, 0])
        assert np.all(relative_action >= 0) and np.all(relative_action <= 1), 'relative_action is not in [0, 1]'
    else:
        assert len(abs_action.shape) == 2, 'abs_action.ndim should be 1 or 2'
        relative_action = abs_action - JOINT_POSITIONS_LIMITS[None, :, 0]
        relative_action = relative_action / (JOINT_POSITIONS_LIMITS[None, :, 1] - JOINT_POSITIONS_LIMITS[None, :, 0])
        assert np.all(relative_action >= 0) and np.all(relative_action <= 1), 'relative_action is not in [0, 1]'

    return relative_action


horizon = 8
# for i in range(len(data)):
#     if i+horizon>=len(data):
#         break
#     abs_actions=data[i:i+horizon]
#     relative_actions=abs_position_2_relative_position(np.array(abs_actions))
#     print(relative_actions)
#     plt.imshow(relative_actions)
#     plt.show()

abs_actions = np.array(data[:80])
relative_actions = abs_position_2_relative_position(abs_actions)
print(relative_actions)
plt.imshow(relative_actions)
plt.show()


print(len(data))


data_path = './1_Data/A_Selfgen/with_path_with_position/train/close_jar_peract/variation0/episodes/episode0/data.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

actions = data['action']

plt.imshow(actions)
plt.show()
