import pickle

episode_path = "/media/jian/ssd4t/zero/1_Data/B_Preprocess/DP_traj/trajectory/test/42/put_groceries_in_cupboard_peract/variation1/episode0/data.pkl"

with open(episode_path, 'rb') as f:
    data = pickle.load(f)
print(data.keys())
