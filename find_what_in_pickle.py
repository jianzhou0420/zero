import pickle


pickle_file = '/media/jian/data/rlbench_frames_0/train/0.pkl'
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
