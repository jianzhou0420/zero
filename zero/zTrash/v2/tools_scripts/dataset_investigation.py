import math
import random
import json
import os
import numpy as np
import pickle
import blosc
import torch


def check_if_keyposes():
    train_path = "/home/jian/git_all/git_manipulation/3d_diffuser_actor/data/peract/Peract_packaged/train"
    files = os.listdir(train_path)

    # 检查数据集是否是1800个dat文件，如果是，说明数据集就是keypose的
    counter = 0
    for folder_path in files:
        dat_files = os.listdir(os.path.join(train_path, folder_path))
        counter += len(dat_files)

    print(counter)
    # 确实是1800,每个episode被缩短成了一个几个keypose的demo


def show_single_dat_file():
    train_path = "/home/jian/git_all/git_manipulation/3d_diffuser_actor/data/peract/Peract_packaged/train"
    files = os.listdir(train_path)


def show_real_trained_pairs():
    train_path = "/home/jian/git_all/git_manipulation/3d_diffuser_actor/data/peract/Peract_packaged/train"
    files = os.listdir(train_path)

    #
    counter = 0
    for folder_path in files:
        dat_files = os.listdir(os.path.join(train_path, folder_path))
        for dat_file in dat_files:
            with open(os.path.join(train_path, folder_path, dat_file), 'rb') as f:
                episode = pickle.loads(blosc.decompress(f.read()))
                max_episode_length = 5
                chunk = random.randint(
                    0, math.ceil(len(episode[0]) / max_episode_length) - 1
                )

                # 以下复杂公式是为了，随机从episode里面选取在max_episode_length范围内的任意一段frame_ids，且尽可能=max_episode_length
                # 这有用么，反正它最后都是
                # Get frame ids for this chunk
                frame_ids = episode[0][chunk * max_episode_length:
                                       (chunk + 1) * max_episode_length
                                       ]

                counter += len(frame_ids)
                print(frame_ids)
                print(counter)
    # counter==8204,一个epoch，总共会跑8204个keypose


def show_num_of_keypose_per_epoch():
    train_path = "/home/jian/git_all/git_manipulation/3d_diffuser_actor/data/peract/Peract_packaged/train"
    files = os.listdir(train_path)

    #
    counter = 0
    for folder_path in files:
        dat_files = os.listdir(os.path.join(train_path, folder_path))
        for dat_file in dat_files:
            with open(os.path.join(train_path, folder_path, dat_file), 'rb') as f:
                data = pickle.loads(blosc.decompress(f.read()))
                counter += len(data[0])
                print(counter)
    # counter==11855,一个epoch，总共有11855个keypose


def find_num_keyposes_larger_than_5():
    train_path = "/home/jian/git_all/git_manipulation/3d_diffuser_actor/data/peract/Peract_packaged/train"
    files = os.listdir(train_path)
    counter = 0
    for folder_path in files:
        dat_files = os.listdir(os.path.join(train_path, folder_path))
        for dat_file in dat_files:
            with open(os.path.join(train_path, folder_path, dat_file), 'rb') as f:
                data = pickle.loads(blosc.decompress(f.read()))
                if len(data[0]) > 5:
                    counter += 1
                    print(counter)


def calculate_avg_keyposes_per_epoch():
    train_path = "/home/jian/git_all/git_manipulation/3d_diffuser_actor/data/peract/Peract_packaged/train"
    files = os.listdir(train_path)
    counter = 0
    for folder_path in files:
        dat_files = os.listdir(os.path.join(train_path, folder_path))
        for dat_file in dat_files:
            with open(os.path.join(train_path, folder_path, dat_file), 'rb') as f:
                data = pickle.loads(blosc.decompress(f.read()))
                if len(data[0]) <= 5:
                    counter += len(data[0])
                else:
                    counter += 5
                print(counter)


show_real_trained_pairs()

# 结论一：3dda实际跑了116,000,000个step
# 所以peract的计算是这样的，总共6个GPU，每个GPU上跑600,000个iteration,
# 每个iteration包含一个bs=8的batch,所以一共28，800，000个episodes，对应文中的16，000个Epochs.
# 此时，这里的epoch是epoch of episodes,它每个episodes会算3-5个keypose,
# 一个epoch1800个episode，约需要算7250个step
# 所以3dda实际算的step为600,000*8*6/1800*7250=116,000,000 step

#
# 由于代码的问题，6张卡和1张卡，用同一个代码，时间会相同，因为计算量直接削掉了N个GPU的计算量。
# 总的计算量是
