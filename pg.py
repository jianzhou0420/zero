import pickle
import os
import sys
import numpy as np


with open('/media/jian/ssd4t/zero/1_Data/A_Selfgen/train/seed42/close_jar/variation0/episodes/episode0/data.pkl', 'rb') as f:
    data = pickle.load(f)


print(data['obs'].shape)
