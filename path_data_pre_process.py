import pickle
sample_data_path = '/media/jian/ssd4t/zero/1_Data/A_Selfgen/0.005all/train/close_jar/variation0/episodes/episode0'

data_pkl = sample_data_path + '/data.pkl'
actions_all_pkl = sample_data_path + '/actions_all.pkl'


with open(data_pkl, 'rb') as f:
    data = pickle.load(f)

with open(actions_all_pkl, 'rb') as f:
    actions_all = pickle.load(f)


print(1)


for key_frame_id, frame_id in enumerate(data['key_frameids']):
    print(data['action'][key_frame_id] == actions_all[frame_id])


preprocess_data_path = '/media/jian/ssd4t/zero/1_Data/B_Preprocess/0.005all/train/close_jar/variation0/episodes/episode0/data.pkl'

with open(preprocess_data_path, 'rb') as f:
    preprocess_data = pickle.load(f)


print(1)
