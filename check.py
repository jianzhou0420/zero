import numpy as np
import pickle
path = '/media/jian/ssd4t/selfgen/seed42/voxel0.004/close_jar/variation0/episodes/episode0/data.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)
print(data['pc'][0].shape)
