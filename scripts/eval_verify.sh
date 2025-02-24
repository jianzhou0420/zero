conda activate zero
# tasks_to_use=("close_jar")
# tasks_to_use=("insert_onto_square_peg" "close_jar")
tasks_to_use=("insert_onto_square_peg" "close_jar" "light_bulb_in" "put_groceries_in_cupboard")
# tasks_to_use=("put_groceries_in_cupboard")
# tasks_to_use=("place_shape_in_shape_sorter")
# python /data/zero/zero/v3/eval_verify.py \
# --config /media/jian/ssd4t/zero/zero/v3/config/after_shock.yaml \
# --name test \
# --checkpoint /data/ckpt/20250122_163025after_shock.yamlepoch=199.ckpt \
# --tasks_to_use ${tasks_to_use[@]} \
# conda activate zero


for item in 1299 799 399 1199 1099
do
    python -m zero.expBaseV5.eval_expbase \
    --config /media/jian/ssd4t/zero/2_Train/2025_02_17__15-20_expBaseV5_4ge/version_0/hparams.yaml\
    --name test \
    --checkpoint /media/jian/ssd4t/zero/2_Train/2025_02_17__15-20_expBaseV5_4ge/version_0/checkpoints/2025_02_17__15-20_expBaseV5_test_epoch=$item.ckpt\
    --tasks_to_use "${tasks_to_use[@]}" \
    --record_video True
done

# tasks_to_use=("place_shape_in_shape_sorter")
# python /data/zero/zero/v3/eval_verify.py \
# --config /media/jian/ssd4t/zero/zero/v3/config/sort_shape_0.005.yaml \
# --name test \
# --checkpoint /media/jian/ssd4t/zero/2_Train/2025_01_23_22:10_sort_shape_0.005/version_0/checkpoints/2025_01_23__22:10_sort_shape_0.005_epoch=1599.ckpt \
# --tasks_to_use ${tasks_to_use[@]} \

