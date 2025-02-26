conda activate zero


tasks_to_use='put_groceries_in_cupboard'




python -m zero.expLongHorizon.eval_LongHorizon \
    --config /media/jian/ssd4t/zero/2_Train/EXP02_24_long_horizon_800_512hidden/version_0/hparams.yaml\
    --name test \
    --checkpoint /media/jian/ssd4t/zero/2_Train/EXP02_24_long_horizon_800_512hidden/version_0/checkpoints/EXP02_24_long_horizon_800_512hidden_epoch=799.ckpt\
    --tasks_to_use "${tasks_to_use[@]}" \

