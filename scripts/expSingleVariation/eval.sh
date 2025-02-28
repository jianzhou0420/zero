conda activate zero


# tasks_to_use='put_groceries_in_cupboard'
tasks_to_use='close_jar'




python -m zero.expLongHorizon.eval_LongHorizon \
    --config /media/jian/ssd4t/zero/2_Train/EXP02_28_single_variation_validataion/version_2/hparams.yaml\
    --name test \
    --checkpoint /media/jian/ssd4t/zero/2_Train/EXP02_28_single_variation_validataion/version_2/checkpoints/EXP02_28_single_variation_validataion_epoch=799.ckpt\
    --tasks_to_use "${tasks_to_use[@]}" \

