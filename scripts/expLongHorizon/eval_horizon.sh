conda activate zero

# tasks_to_use='put_groceries_in_cupboard'
tasks_to_use='close_jar'


exp_dir=$1

python -m zero.expLongHorizon.eval_LongHorizon \
    --exp-config /media/jian/ssd4t/zero/zero/expLongHorizon/config/eval.yaml\
    exp_dir $exp_dir \
    headless True \
    num_workers 4 \
    tasks_to_use $tasks_to_use \
    

