module load  Singularity/3.10.5
export SINGULARITY_IMAGE_PATH='/data/singularity/nvcuda_v2.sif'
export python_bin='/data/conda_env/zero/bin/python'


# singularity shell --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH


singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH xvfb-run -a ${python_bin} -m zero.expBaseV5.trainer_expbase \
      