

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
file_path_1 = 'results_visualization/insert_peg_0.01/results.csv'
file_path_2 = 'results_visualization/insert_peg_0.005/results.csv'

with open(file_path_1, 'r') as f:
    df = pd.read_csv(f)

epoch_number1 = []
sr1 = []
for index, row in df.iterrows():
    epoch_number1.append(row['ckpt_name'].split('=')[-1].split('.')[0])
    sr1.append(float(row['sr'].split('[')[-1].split(']')[0]))

with open(file_path_2, 'r') as f:
    df = pd.read_csv(f)

epoch_number2 = []
sr2 = []
for index, row in df.iterrows():
    epoch_number2.append(row['ckpt_name'].split('=')[-1].split('.')[0])
    sr2.append(float(row['sr'].split('[')[-1].split(']')[0]))

# print(epoch_number1)

print(sr1, sr2)
print(epoch_number1, epoch_number2)
x = np.arange(0, 2000, 100)
y1 = np.full_like(x, np.nan, dtype=np.float16)

for i, epoch in enumerate(epoch_number1):
    y1[int(round(int(epoch) + 1) / 100)] = sr1[i]
