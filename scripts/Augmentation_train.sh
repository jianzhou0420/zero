# conda activate /data/conda_env/zero
tasks_to_use='close_jar'
# tasks_to_use='insert_onto_square_peg'
# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expAugmentation.trainer_expbase \
        --exp-config /data/zero/zero/expAugmentation/config/expBase_Lotus.yaml \
        name EXP03_04_augmentation_valid\
        dataset augment\
        num_gpus 1 \
        epoches 400 \
        batch_size 4 \
        TRAIN_DATASET.num_points 4096 \
        TRAIN_DATASET.pos_bins 15 \
        TRAIN_DATASET.pos_bin_size 0.01 \
        MODEL.action_config.pos_bins 15 \
        MODEL.action_config.pos_bin_size 0.01 \
        MODEL.action_config.voxel_size 0.01\
        tasks_to_use $tasks_to_use \
        TRAIN.n_workers 4\
        B_Preprocess /media/jian/ssd4t/zero/1_Data/B_Preprocess/0.01all_with_path\
        des "debug"\
        
        # horizon 1 \
        # MODEL.action_config.horizon 1 \

