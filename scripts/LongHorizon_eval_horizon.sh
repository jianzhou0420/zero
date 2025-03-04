conda activate zero

# tasks_to_use='put_groceries_in_cupboard'
# tasks_to_use="insert_onto_square_peg"
tasks_to_use="close_jar"


exp_dir=/data/zero/2_Train/EXP03_03_close_jar/version_0

python -m zero.expLongHorizon.eval_LongHorizon \
    --exp-config /data/zero/zero/expLongHorizon/config/eval.yaml\
    exp_dir $exp_dir \
    headless True \
    num_workers 4 \
    tasks_to_use $tasks_to_use \
    

