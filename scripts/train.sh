conda activate zero

python  -m zero.expBaseV5.trainer_expbase \
        --exp-config /data/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        name expBaseV5_test \
        dataset augment\
        num_gpus 1 \
        epoches 1200 \
        batch_size 4 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001\
        MODEL.action_config.pos_bins 75\
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use "['insert_onto_square_peg','close_jar','light_bulb_in','put_groceries_in_cupboard']" \
        fp16 True \
        
        



