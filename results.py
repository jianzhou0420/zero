import csv
import os
import json
import re

from zero.v3.dataprocess.utils import natural_sort_key


eval_dir = '/data/zero/3_Eval/eval_log'
dir_list = sorted(os.listdir(eval_dir), key=natural_sort_key)


def get_results(data):
    total_demos = 0
    total_success = 0
    for line in data:
        checkpoint = line['checkpoint'].split('/')[-1]
        task_name = line['task']
        variation = line['variation']
        num_demos = line['num_demos']
        sr = line['sr']
        total_demos += num_demos
        total_success += round(sr * num_demos)
        sr = total_success / total_demos
    return checkpoint, task_name, total_demos, total_success, sr


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
    result_path = os.path.join(eval_dir, dir_name, 'preds/seed42/', 'results.jsonl')
    if not os.path.exists(result_path):
        if not os.path.exists(os.path.join(eval_dir, dir_name, 'preds/preds/seed42/', 'results.jsonl')):
            continue
        else:
            result_path = os.path.join(eval_dir, dir_name, 'preds/preds/seed42/', 'results.jsonl')
    data = []
    with open(result_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    checkpoint, task_name, total_demos, total_success, sr = get_results(data)
    outs_dict['ckpt_name'].append(checkpoint)
    outs_dict['task_name'].append(task_name)
    outs_dict['total_demos'].append(total_demos)
    outs_dict['total_success'].append(total_success)
    outs_dict['sr'].append(sr)
    outs.append(outs_dict)

with open('results.csv', 'w', newline='') as csvfile:
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
