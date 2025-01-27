
conda activate zero

# name=insert_peg_0.01
# config_path=/media/jian/ssd4t/zero/zero/v3/config/insert_peg_0.01.yaml

# python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name


name=sort_shape_edge
config_path=/media/jian/ssd4t/zero/zero/v3/config/sort_shape_edge.yaml

python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name
