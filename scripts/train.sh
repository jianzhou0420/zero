
conda activate zero

# name=insert_peg_0.01
# config_path=/media/jian/ssd4t/zero/zero/v3/config/insert_peg_0.01.yaml

# python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name


name=0.005_sort_shape
config_path=/data/zero/zero/v3/config/0.005_sort_shape.yaml

python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name
