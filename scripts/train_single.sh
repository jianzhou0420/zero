
conda activate zero

name=expBins_all_together_single_test
config_path=/media/jian/ssd4t/zero/zero/expBaseV4/config/expBase_all_together_single_test.yaml

python /media/jian/ssd4t/zero/zero/expBins/trainer_expbase.py --config $config_path --name $name


