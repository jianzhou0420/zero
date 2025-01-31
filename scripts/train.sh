
conda activate zero

# name=insert_peg_0.01
# config_path=/media/jian/ssd4t/zero/zero/v3/config/insert_peg_0.01.yaml

# python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name


name=exp_12_bins_insert_peg_0.005
config_path=/data/zero/zero/v4/config/insert_peg_0.005_bins.yaml

python /data/zero/zero/v4/trainer_lotus.py --config $config_path --name $name
