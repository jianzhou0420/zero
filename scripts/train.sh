
conda activate zero

name=expBase_4gpu_1_batch_size
config_path=/data/zero/zero/expAugmentation/config/expBase_all_together.yaml

python /data/zero/zero/expAugmentation/trainer_expbase.py --config $config_path --name $name --num_gpu 4 --epoches 800 --batch_size 1


