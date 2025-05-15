from zero.tmp.replay_buffer import ReplayBuffer


test_buffer = ReplayBuffer.create_from_path('/media/jian/ssd4t/zero/1_Data/B_Preprocess/zarr/DP_traj_zarr/trajectory/reach_target')
print(test_buffer.episode_lengths)
