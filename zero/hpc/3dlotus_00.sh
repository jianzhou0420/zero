#!/bin/bash

# Configure the resources required
#SBATCH -p a100                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1          
#SBATCH -c 4    	                          
#SBATCH --gres=gpu:1           	                             
#SBATCH --time=1-23:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour           
#SBATCH --mem=60GB                                              # specify memory required per node (here set to 16 GB)


# Configure log
#SBATCH --output=/hpcfs/users/a1946536/%j.out
#SBATCH --error=/hpcfs/users/a1946536/%j.out

# Configure notifications 
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=a1946536@adelaide.edu.au          # Email to which notifications will be sent

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
module purge
module load CUDA/11.6.2
source ~/.bashrc 
conda activate test2
conda env list
export DATA_HOME='/hpcfs/users/a1946536/lotus'
export CODE_HOME='/hpcfs/users/a1946536/code/'

nvidia-smi
cd /hpcfs/users/a1946536/code/zero/
srun python zero/v1/trainer_lotus.py

