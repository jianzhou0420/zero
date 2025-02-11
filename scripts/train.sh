
conda activate zero

name=expBins_all_together
config_path=/data/zero/zero/expAugmentation/config/expBase_all_together.yaml

python /data/zero/zero/expAugmentation/trainer_expbase.py --config $config_path --name $name --num_gpu 4


