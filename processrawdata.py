from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import lmdb
import os
import msgpack
# Path to your LMDB database
import re
from codebase.Tools.PointCloudDrawer import PointCloudDrawer
import numpy as np

pcdrawer = PointCloudDrawer()


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


root_path = "/media/jian/ssd4t/selfgen/train_dataset/keysteps/seed42"

# sort with nature languageorder
all_dirs = sorted(os.listdir(root_path), key=natural_sort_key)


def read_convert(rgb, frame_idx):
    data = rgb[b'data']
    type = rgb[b'type']
    shape = rgb[b'shape']

    rgb = np.frombuffer(data, dtype=type)
    rgb = rgb.reshape(shape)
    rgb = rgb[frame_idx]
    rgb = rgb.reshape(-1, 3)
    return rgb


len_all_dirs = len(all_dirs)
for i in tqdm(range(len_all_dirs)):
    this_dir = all_dirs[i]
    this_task, variation = this_dir.split("+")
    print('showing', this_task, variation)
    # if this_task != "place_shape_in_shape_sorter_peract":
    #     continue
    lmdb_path = os.path.join(root_path, this_dir)
# Open the database (readonly=False for write access)
    env = lmdb.open(lmdb_path, readonly=True)
    txn = env.begin()
    test = [key for key in txn.cursor().iternext(values=False)]

    values = [txn.get(key) for key in test]
    values = [msgpack.unpackb(value) for value in values]
    for episode_idx, single_episode in enumerate(values):
        try:
            rgb = single_episode['rgb']
            pc = single_episode['pc']
        except:
            continue

        for frame_idx, frame in enumerate(single_episode['key_frameids']):
            this_rgb = read_convert(rgb, frame_idx)

            this_pc = read_convert(pc, frame_idx)

            this_rbg = this_rgb.reshape(-1, 512, 512, 3)
            plt.imshow(this_rbg[1])
            plt.show()
            # pcdrawer.save_onece(this_pc, colors=this_rgb, save_path=f"/media/jian/ssd4t/selfgen/output1/{this_task}+{variation}/{episode_idx}/{frame_idx}.html")
            break
        break
    # print(test)
