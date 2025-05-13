#!/bin/bash

# Configure the resources required
#SBATCH -p a100                                                # partition (this is the queue your job will be added to)
#SBATCH --job-name=test1
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH --cpus-per-task=16             	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=04:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=64GB

source ~/.bashrc
conda activate zero

echo "Running task $SLURM_ARRAY_TASK_ID"

# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python -m zero.expBaseV5.trainer_expbase \
    --exp-config ./zero/expBaseV5/config/expBase_Lotus.yaml \
    name 150bins dataset augment num_gpus 1 \
    epoches 1200 \
    batch_size 4 \
    TRAIN_DATASET.num_points 100000 \
    TRAIN_DATASET.pos_bins 150 \
    TRAIN_DATASET.pos_bin_size 0.005 MODEL.action_config.pos_bins 150 MODEL.action_config.pos_bin_size 0.005 \
    tasks_to_use "$tasks_to_use"
