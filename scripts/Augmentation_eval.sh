conda activate zero


tasks_to_use="insert_onto_square_peg"



exp_dir=/data/zero/2_Train/2025_03_04__server/version_0

python -m zero.expAugmentation.eval_LongHorizon \
    --exp-config /data/zero/zero/expAugmentation/config/eval.yaml\
    exp_dir $exp_dir \
    headless True \
    num_workers 4 \
    tasks_to_use "$tasks_to_use" \
    record_video True \
    num_demos 10

    

