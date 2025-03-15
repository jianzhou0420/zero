import pickle

with open('/data/zero/1_Data/B_Preprocess/0.005all_with_path_with_positionactions/train/close_jar/variation0/episodes/episode0/data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
