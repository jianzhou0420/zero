# conda activate /data/conda_env/zero

conda activate zero

# tasks_to_use='close_jar'
tasks_to_use='insert_onto_square_peg'
# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expLongHorizon.trainer_expbase \
        --exp-config /data/zero/zero/expLongHorizon/config/expBase_Lotus.yaml \
        name EXP03_02_insert\
        dataset augment\
        num_gpus 1 \
        epoches 800 \
        batch_size 4 \
        TRAIN_DATASET.num_points 409600 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001 \
        MODEL.action_config.pos_bins 75 \
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use $tasks_to_use \
        TRAIN.n_workers 4\
        MODEL.action_config.action_head_type multihead \
        B_Preprocess /media/jian/ssd4t/zero/1_Data/B_Preprocess/0.005all \
        des "debug"\
        
        # horizon 1 \
        # MODEL.action_config.horizon 1 \

