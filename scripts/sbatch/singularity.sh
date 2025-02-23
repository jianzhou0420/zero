module load  Singularity/3.10.5
export SINGULARITY_IMAGE_PATH='/hpcfs/users/a1946536/singularity/nvcuda_v2.sif'
export python_bin='/hpcfs/users/a1946536/conda_env/zero/bin/python'


# singularity shell --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH


singularity exec --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv $SINGULARITY_IMAGE_PATH xvfb-run -a ${python_bin} -m zero.expBaseV5.trainer_expbase \
        --exp-config /hpcfs/users/a1946536/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        name $exp_name\
        dataset augment\
        num_gpus 1 \
        epoches 800 \
        batch_size 4 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001\
        MODEL.action_config.pos_bins 75\
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use "$tasks_to_use" \