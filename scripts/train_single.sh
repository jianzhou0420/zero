
conda activate zero

name=expBins_all_together_single_test
config_path=/media/jian/ssd4t/zero/zero/expAugmentation/config/expBase_all_together_single_test.yaml

python /media/jian/ssd4t/zero/zero/expAugmentation/trainer_expbase.py --config $config_path --name $name


