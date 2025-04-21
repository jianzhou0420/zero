import torch
import torch.nn as nn
import pickle


with open('/media/jian/ssd4t/zero/1_Data/B_Preprocess/DA3D/put_groceries_in_cupboard/variation0/episodes/episode0/data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
# eePose = data['eePose_hist'][0][0, :]
# print(eePose.shape)
# print(eePose)

# quat = eePose[3:7]
print(data['rgb'][0].shape)
