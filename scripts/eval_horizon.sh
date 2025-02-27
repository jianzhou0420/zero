conda activate zero


tasks_to_use='put_groceries_in_cupboard'
# tasks_to_use='close_jar'




python -m zero.expLongHorizon.eval_LongHorizon \
    --config /media/jian/ssd4t/zero/2_Train/EXP02_26_multihead/version_0/hparams.yaml\
    --name test \
    --checkpoint /media/jian/ssd4t/zero/2_Train/EXP02_26_multihead/version_0/checkpoints/EXP02_26_multihead_epoch=799.ckpt \
    --tasks_to_use "${tasks_to_use[@]}" \

