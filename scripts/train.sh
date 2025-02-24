# conda activate /data/conda_env/zero

conda activate zero

# 中等
if [ "$1" == "medium" ]; then
    tasks_to_use="insert_onto_square_peg,close_jar,light_bulb_in,put_groceries_in_cupboard"
elif [ "$1" == "small" ]; then
    tasks_to_use="insert_onto_square_peg,close_jar"
elif [ "$1" == "large" ]; then
    tasks_to_use=""  # Empty array for "large"
elif [ "$1" == "single" ]; then
    tasks_to_use="insert_onto_square_peg"
fi


# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expBaseV5.trainer_expbase \
        --exp-config /data/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        name EXP02_24_bins_experimental_group \
        dataset augment\
        num_gpus 1 \
        epoches 800 \
        batch_size 4 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 150 \
        TRAIN_DATASET.pos_bin_size 0.005\
        MODEL.action_config.pos_bins 150\
        MODEL.action_config.pos_bin_size 0.005 \
        tasks_to_use $tasks_to_use \
       
        
        
#TODO:配置没有显示器的eval，有空再说


python  -m zero.expBaseV5.trainer_expbase \
        --exp-config /data/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        name EXP02_24_bins_control_group \
        dataset augment\
        num_gpus 1 \
        epoches 800 \
        batch_size 4 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001\
        MODEL.action_config.pos_bins 75\
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use $tasks_to_use \
       
        