#!/bin/bash -l
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --array=1-3
#SBATCH --error="results/tsp_%a.err"
#SBATCH --output="results/tsp_%a.out"
#SBATCH --job-name="tsp_ga"

module load Anaconda3/2020.07
conda activate /apps/examples/envs/arraytut

echo "array job index: $SLURM_ARRAY_TASK_ID"
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p parameters.csv`

python tsp_ga.py --input sample.tsp --seed ${par[0]} --mutation ${par[1]} --generation ${par[2]} --population ${par[3]}