# conda activate /data/conda_env/zero

conda activate zero

tasks_to_use='close_jar'
# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expSingleVariation.trainer_expbase \
        --exp-config /data/zero/zero/expSingleVariation/config/expBase_Lotus.yaml \
        name EXP02_28_single_variation_validataion\
        dataset augment\
        num_gpus 1 \
        epoches 1600 \
        batch_size 5 \
        TRAIN_DATASET.num_points 4096 \
        TRAIN_DATASET.pos_bins 15 \
        TRAIN_DATASET.pos_bin_size 0.005\
        MODEL.action_config.pos_bins 15\
        MODEL.action_config.pos_bin_size 0.005\
        tasks_to_use $tasks_to_use \
        TRAIN.n_workers 4\
        horizon 1  \
        MODEL.action_config.horizon 1 \
        MODEL.action_config.action_head_type multihead \
        TRAIN_DATASET.variations_to_use '[0]' \
        TRAIN.num_train_steps 1600\
        TRAIN.warmup_steps 200\
       
# python  -m zero.expLongHorizon.trainer_expbase \
#         --exp-config /data/zero/zero/expLongHorizon/config/expBase_Lotus.yaml \
#         name EXP02_24_long_horizon_1200 \
#         dataset augment\
#         num_gpus 1 \
#         epoches 1200\
#         batch_size 4 \
#         TRAIN_DATASET.num_points 4096 \
#         TRAIN_DATASET.pos_bins 15 \
#         TRAIN_DATASET.pos_bin_size 0.005\
#         MODEL.action_config.pos_bins 15\
#         MODEL.action_config.pos_bin_size 0.005 \
#         tasks_to_use $tasks_to_use \
#         TRAIN.n_workers 4
       
       
        
# #TODO:配置没有显示器的eval，有空再说


# python  -m zero.expBaseV5.trainer_expbase \
#         --exp-config /data/zero/zero/expBaseV5/config/expBase_Lotus.yaml \
#         name EXP02_24_bins_control_group \
#         dataset augment\
#         num_gpus 1 \
#         epoches 800 \
#         batch_size 4 \
#         TRAIN_DATASET.num_points 100000 \
#         TRAIN_DATASET.pos_bins 75 \
#         TRAIN_DATASET.pos_bin_size 0.001\
#         MODEL.action_config.pos_bins 75\
#         MODEL.action_config.pos_bin_size 0.001 \
#         tasks_to_use $tasks_to_use \
       
        