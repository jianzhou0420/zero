import pickle
single_episode_path = '/data/zero/data/selfgen/seed42/place_shape_in_shape_sorter/variation0/episodes/episode0/data.pkl'

with open(single_episode_path, 'rb') as f:
    data = pickle.load(f)
