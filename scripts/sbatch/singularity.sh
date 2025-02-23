module load  Singularity/3.10.5
export SINGULARITY_IMAGE_PATH='/hpcfs/users/a1946536/singularity/nvcuda_v2.sif'
export python_bin='/hpcfs/users/a1946536/conda_env/zero/bin/python'


# singularity shell --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH


singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH xvfb-run -a ${python_bin} -m zero.expBaseV5.trainer_expbase \
      