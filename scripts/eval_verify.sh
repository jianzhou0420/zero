conda activate zero
tasks_to_use=("close_jar")
# tasks_to_use=("insert_onto_square_peg")
# tasks_to_use=("insert_onto_square_peg" "close_jar" "light_bulb_in" "put_groceries_in_cupboard")

# tasks_to_use=("put_groceries_in_cupboard")
# tasks_to_use=("place_shape_in_shape_sorter")
# python ./zero/v3/eval_verify.py \
# --config ./zero/v3/config/after_shock.yaml \
# --name test \
# --checkpoint /data/ckpt/20250122_163025after_shock.yamlepoch=199.ckpt \
# --tasks_to_use ${tasks_to_use[@]} \
# conda activate zero

for item in 1299; do
    python -m zero.expBaseV5.eval_expbase \
        --config ./2_Train/2025_03_03__13-03_EXP03_03_rollback_change_datasettowipath/version_0/hparams.yaml --name test \
        --checkpoint ./2_Train/2025_03_03__13-03_EXP03_03_rollback_change_datasettowipath/version_0/checkpoints/2025_03_03__13-03_EXP03_03_rollback_change_datasettowipath_epoch=399.ckpt --tasks_to_use "${tasks_to_use[@]}"

done

# tasks_to_use=("place_shape_in_shape_sorter")
# python ./zero/v3/eval_verify.py \
# --config ./zero/v3/config/sort_shape_0.005.yaml \
# --name test \
# --checkpoint ./2_Train/2025_01_23_22:10_sort_shape_0.005/version_0/checkpoints/2025_01_23__22:10_sort_shape_0.005_epoch=1599.ckpt \
# --tasks_to_use ${tasks_to_use[@]} \
