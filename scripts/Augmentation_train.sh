# conda activate /data/conda_env/zero
# tasks_to_use='close_jar'
tasks_to_use='insert_onto_square_peg,close_jar,put_groceries_in_cupboard,place_cups'
# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")




python  -m zero.expForwardKinematics.trainer_expbase \
        --exp-config /data/zero/zero/expForwardKinematics/config/expBase_Lotus.yaml \
        name EXP03_04_insert_close_jar_0.005\
        dataset augment\
        num_gpus 1 \
        epoches 1800 \
        batch_size 1 \
        TRAIN_DATASET.num_points 4096 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001 \
        TRAIN_DATASET.euler_resolution 1 \
        MODEL.action_config.pos_bins 75 \
        MODEL.action_config.pos_bin_size 0.001 \
        MODEL.action_config.voxel_size 0.005\
        MODEL.action_config.euler_resolution 1\
        tasks_to_use "$tasks_to_use" \
        TRAIN.n_workers 4\
        B_Preprocess /data/zero/1_Data/B_Preprocess/0.005all \
        des "to see close_jar and insert at 0.005"\
        
        # horizon 1 \
        # MODEL.action_config.horizon 1 \

