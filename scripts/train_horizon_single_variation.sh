# conda activate /data/conda_env/zero

conda activate zero

tasks_to_use='put_groceries_in_cupboard'
# tasks_to_use=("meat_off_grill" "sweep_to_dustpan_of_size" "close_jar" "push_buttons" "light_bulb_in" "insert_onto_square_peg" "put_groceries_in_cupboard" "place_shape_in_shape_sorter" "stack_blocks")

python  -m zero.expLongHorizon.trainer_expbase \
        --exp-config /data/zero/zero/expLongHorizon/config/expBase_Lotus.yaml \
        name EXP02_26_multihead\
        dataset augment\
        num_gpus 1 \
        epoches 800 \
        batch_size 1 \
        TRAIN_DATASET.num_points 4096 \
        TRAIN_DATASET.pos_bins 15 \
        TRAIN_DATASET.pos_bin_size 0.005\
        MODEL.action_config.pos_bins 15\
        MODEL.action_config.pos_bin_size 0.005 \
        tasks_to_use $tasks_to_use \
        TRAIN.n_workers 4\
        MODEL.ptv3_config.dec_channels "[128, 128, 256, 512]" \
        TRAIN_DATASET.variations_to_use "[0]" \
        TRAIN.num_train_steps 9600\
        TRAIN.warmup_steps 500\
        
       
