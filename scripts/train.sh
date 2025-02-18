conda activate zero

python  -m zero.expBaseV5.trainer_expbase \
        --exp-config /data/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        name expBaseV5_test \
        dataset augment\
        num_gpus 1 \
        epoches 1200 \
        batch_size 2 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001\
        MODEL.action_config.pos_bins 75\
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use "[meat_off_grill, sweep_to_dustpan_of_size, close_jar, push_buttons, light_bulb_in, insert_onto_square_peg, put_groceries_in_cupboard,place_shape_in_shape_sorter,stack_blocks]" \
       
        
        



