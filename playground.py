import lmdb
import os
import msgpack
import pickle
# Path to your LMDB database
path = '/media/jian/ssd4t/selfgen/20250105/train_dataset/post_process_keysteps/seed42/voxel0.001/close_jar/variation0/episodes/episode0/data.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
print(data.keys())
