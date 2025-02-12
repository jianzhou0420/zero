conda activate zero
# tasks_to_use=("close_jar")
tasks_to_use=("insert_onto_square_peg")
# tasks_to_use=("place_shape_in_shape_sorter")
# python /data/zero/zero/v3/eval_verify.py \
# --config /media/jian/ssd4t/zero/zero/v3/config/after_shock.yaml \
# --name test \
# --checkpoint /data/ckpt/20250122_163025after_shock.yamlepoch=199.ckpt \
# --tasks_to_use ${tasks_to_use[@]} \

# conda activate zero


for item in 799
do

    python /media/jian/ssd4t/zero/zero/expAugmentation/eval_expbase.py \
    --config /media/jian/ssd4t/zero/2_Train/zzzz/version_0/hparams.yaml \
    --name test \
    --checkpoint /data/zero/2_Train/zzzz/version_0/checkpoints/2025_02_11__10-29_expBins_all_together_epoch=799.ckpt\
    --tasks_to_use ${tasks_to_use[@]} 

done

# tasks_to_use=("place_shape_in_shape_sorter")
# python /data/zero/zero/v3/eval_verify.py \
# --config /media/jian/ssd4t/zero/zero/v3/config/sort_shape_0.005.yaml \
# --name test \
# --checkpoint /media/jian/ssd4t/zero/2_Train/2025_01_23_22:10_sort_shape_0.005/version_0/checkpoints/2025_01_23__22:10_sort_shape_0.005_epoch=1599.ckpt \
# --tasks_to_use ${tasks_to_use[@]} \

