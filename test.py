
import pickle

this_path = "/data/zero/1_Data/B_Preprocess/DA3D/close_jar/variation0/episodes/episode0/data.pkl"

with open(this_path, 'rb') as f:
    data = pickle.load(f)
print(data.keys())
