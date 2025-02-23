conda activate /hpcfs/users/a1946536/conda_env/zero

exp_name=$1

# 中等
if [ "$2" == "medium" ]; then
    tasks_to_use="insert_onto_square_peg,close_jar,light_bulb_in,put_groceries_in_cupboard"
elif [ "$2" == "small" ]; then
    tasks_to_use="insert_onto_square_peg,close_jar"
elif [ "$2" == "large" ]; then
    tasks_to_use=""  # Empty array for "large"
elif [ "$2" == "single" ]; then
    tasks_to_use="insert_onto_square_peg"
fi


# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expBaseV5.trainer_expbase \
        --exp-config /hpcfs/users/a1946536/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
        name $exp_name \
        dataset augment\
        num_gpus 1 \
        epoches 800 \
        batch_size 1 \
        TRAIN_DATASET.num_points 100000 \
        TRAIN_DATASET.pos_bins 75 \
        TRAIN_DATASET.pos_bin_size 0.001\
        MODEL.action_config.pos_bins 75\
        MODEL.action_config.pos_bin_size 0.001 \
        tasks_to_use $tasks_to_use \
       
        
        



