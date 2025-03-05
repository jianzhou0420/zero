import csv
import os
import json
import re

from zero.dataprocess.utils import natural_sort_key


eval_dir = '/data/zero/3_Eval/eval_log'
dir_list = sorted(os.listdir(eval_dir), key=natural_sort_key)


def get_results(data):
    tasks = []
    for line in data:
        if line['task'] not in tasks:
            tasks.append(line['task'])
    checkpoint_list = []
    task_name_list = []
    total_demos_list = []
    total_success_list = []
    sr_list = []
    for task in tasks:
        total_demos = 0
        total_success = 0
        for line in data:
            if line['task'] != task:
                continue
            checkpoint = line['checkpoint'].split('/')[-1]
            variation = line['variation']
            num_demos = line['num_demos']
            sr = line['sr']
            total_demos += num_demos
            total_success += round(sr * num_demos)
            sr = total_success / total_demos

        checkpoint_list.append(checkpoint)
        task_name_list.append(task)
        total_demos_list.append(total_demos)
        total_success_list.append(total_success)
        sr_list.append(sr)
    return checkpoint_list, task_name_list, total_demos_list, total_success_list, sr_list


outs = []

sr_list = []
x_list = []
for dir_name in dir_list:

    outs_dict = {
        'ckpt_name': [],
        'task_name': [],
        'total_demos': [],
        'total_success': [],
        'sr': []
    }
    seed = os.listdir(os.path.join(eval_dir, dir_name, 'preds/'))[0].split('seed')[-1]
    seed = 'seed' + seed
    result_path = os.path.join(eval_dir, dir_name, 'preds/', seed, 'results.jsonl')
    if not os.path.exists(result_path):
        if not os.path.exists(os.path.join(eval_dir, dir_name, 'preds/preds/', seed, 'results.jsonl')):
            continue
        else:
            result_path = os.path.join(eval_dir, dir_name, 'preds/preds/', seed, 'results.jsonl')
    data = []
    with open(result_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    checkpoint, task_name, total_demos, total_success, sr = get_results(data)
    num_tasks = len(task_name)
    for i in range(num_tasks):
        outs_dict['ckpt_name'].append(checkpoint[i])
        outs_dict['task_name'].append(task_name[i])
        outs_dict['total_demos'].append(total_demos[i])
        outs_dict['total_success'].append(total_success[i])
        outs_dict['sr'].append(sr[i])
        outs.append(outs_dict)
        outs_dict = {
            'ckpt_name': [],
            'task_name': [],
            'total_demos': [],
            'total_success': [],
            'sr': []
        }

with open('/data/zero/results.csv', 'w', newline='') as csvfile:
    fieldnames = ['ckpt_name', 'task_name', 'total_demos', 'total_success', 'sr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # sort the according to the checkpoint name

    # def natural_sort_key_temp(s):
    #     return [text.lower() for text in re.split('([0-9]+)', s['ckpt_name'][0])]
    # outs = sorted(outs, key=natural_sort_key_temp)
    writer.writeheader()
    for i, out in enumerate(outs):
        writer.writerow({
            'ckpt_name': out['ckpt_name'][0],
            'task_name': out['task_name'][0],
            'total_demos': out['total_demos'][0],
            'total_success': out['total_success'][0],
            'sr': out['sr']
        })
