import numpy as np
import json

import os
this_path = '/media/jian/ssd4t/z_Trash/exp/exp0_verification/preds'
# only folders
all_seeds = [f for f in os.listdir(this_path) if os.path.isdir(os.path.join(this_path, f))]
print(all_seeds)


tmp_dict = {}
variation_all = np.zeros((9, 2))
for seed in all_seeds:
    json_seed_path = os.path.join(this_path, seed, 'results.jsonl')
    with open(json_seed_path, 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            if line_dict['task'] == 'put_groceries_in_cupboard_peract':
                variation = line_dict['variation']
                num_demos = line_dict['num_demos']
                sr = line_dict['sr']
                num_success = num_demos * sr
                variation_all[variation, 0] += num_demos
                variation_all[variation, 1] += num_success

test = variation_all[:, 1] / variation_all[:, 0]
print(test)
for variation in range(9):
    print('variation:', variation, 'num_demos:', variation_all[variation, 0], 'num_success:', variation_all[variation, 1], 'sr:', test[variation])
