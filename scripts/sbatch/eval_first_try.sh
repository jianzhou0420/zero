#!/bin/bash

# Configure the resources required
#SBATCH -p a100                                                # partition (this is the queue your job will be added to)
#SBATCH --job-name=test1
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH --cpus-per-task=36          	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=01:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=128GB

module purge
source ~/.bashrc
cd zero
# module load CUDA/11.6.2
# module load OpenSSL/1.1.1k-GCCcore-11.2.0
# module load Mesa/21.1.7-GCCcore-11.2.0
module load Singularity/3.10.5
conda activate /data/conda_env/zero

export SINGULARITY_IMAGE_PATH='/data/singularity/nvcuda_v2.sif'
export python_bin='/data/conda_env/zero/bin/python'
tasks_to_use=("insert_onto_square_peg")
export SINGULARITYENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}

export XDG_RUNTIME_DIR=$SCRATCH/tmp/runtime-$SLURM_JOBID

mkdir -p $XDG_RUNTIME_DIR
chmod 700 $XDG_RUNTIME_DIR
# singularity shell --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH

echo $LD_LIBRARY_PATH
singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH xvfb-run -a ${python_bin} -m zero.expBaseV5.eval_expbase \
    --config ./2_Train/2025_02_22__15-35_single_try/version_0/hparams.yaml --name test \
    --checkpoint ./2_Train/2025_02_22__15-35_single_try/version_0/checkpoints/2025_02_22__15-35_single_try_epoch=799.ckpt --tasks_to_use ${tasks_to_use[@]} \
    --record_video False
