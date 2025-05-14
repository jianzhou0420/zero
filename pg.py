from zero.tmp.replay_buffer import ReplayBuffer

from zero.tmp.sampler import SequenceSampler

import pdb
test = ReplayBuffer.copy_from_path(
    '/media/jian/ssd4t/zero/1_Data/B_Preprocess/zarr/DP_traj_zarr/trajectory/pusht_cchi_v7_replay.zarr')

print(test.get_episode(0).keys())

pdb.set_trace()
