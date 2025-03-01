# conda activate /data/conda_env/zero

conda activate zero

tasks_to_use='close_jar'
# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expLongHorizon.trainer_expbase \
        --exp-config /data/zero/zero/expLongHorizon/config/expBase_Lotus.yaml \
        name EXP02_26_multihead_val_for_0.01_with_horizon_8\
        dataset augment\
        num_gpus 1 \
        epoches 200 \
        batch_size 4 \
        TRAIN_DATASET.num_points 4096 \
        TRAIN_DATASET.pos_bins 30 \
        TRAIN_DATASET.pos_bin_size 0.01 \
        MODEL.action_config.pos_bins 30 \
        MODEL.action_config.pos_bin_size 0.01 \
        tasks_to_use $tasks_to_use \
        TRAIN.n_workers 4\
        horizon 8 \
        MODEL.action_config.horizon 8 \
        MODEL.action_config.action_head_type multihead \
        B_Preprocess /media/jian/ssd4t/zero/1_Data/B_Preprocess/0.01all_with_path \
        des "Validation on multihead horizon=8 close_jar 0.01 voxels, in order to fast validation "


python  -m zero.expLongHorizon.trainer_expbase \
        --exp-config /data/zero/zero/expLongHorizon/config/expBase_Lotus.yaml \
        name EXP02_26_multihead_val_for_0.01_with_horizon_8\
        dataset augment\
        num_gpus 1 \
        epoches 200 \
        batch_size 4 \
        TRAIN_DATASET.num_points 4096 \
        TRAIN_DATASET.pos_bins 30 \
        TRAIN_DATASET.pos_bin_size 0.01 \
        MODEL.action_config.pos_bins 30 \
        MODEL.action_config.pos_bin_size 0.01 \
        tasks_to_use $tasks_to_use \
        TRAIN.n_workers 4\
        horizon 8 \
        MODEL.action_config.horizon 8 \
        MODEL.action_config.action_head_type multihead \
        B_Preprocess /media/jian/ssd4t/zero/1_Data/B_Preprocess/0.01all_with_path \
        des "Validation on multihead horizon=8 close_jar 0.01 voxels, in order to fast validation "




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
       
        