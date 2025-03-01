conda activate zero


# tasks_to_use='put_groceries_in_cupboard'
tasks_to_use='close_jar'


exp_dir='/media/jian/ssd4t/zero/2_Train/EXP02_28_single_variation_validataion/version_3'

python -m zero.expSingleVariation.eval_LongHorizon \
    --exp-config /media/jian/ssd4t/zero/zero/expLongHorizon/config/eval.yaml\
    exp_dir $exp_dir \
    # epoch 1599 \
    

