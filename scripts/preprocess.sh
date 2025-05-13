conda activate zero

python -m zero.dataprocess.ObsProcessor \
        --exp-config ./zero/expBaseV5/config/expBase_Lotus.yaml \
        resume True \
        name expBaseV5_test \
        dataset augment num_gpus 1 \
        epoches 1200 \
        batch_size 2 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001 MODEL.action_config.pos_bins 75 MODEL.action_config.pos_bin_size 0.001
