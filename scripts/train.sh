
conda activate zero

name=insert_peg_0.01
config_path=/media/jian/ssd4t/zero/zero/v4/config/insert_peg_0.01.yaml

python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name


name=exp13_insert_peg_0.005_scale_bins
config_path=/media/jian/ssd4t/zero/zero/v4/config/expbase_voxel_size_0.005.yaml
python /data/zero/zero/v4/trainer_lotus.py --config $config_path --name $name

# name=exp14_insert_peg_0.005_plain_global_bins
# config_path=/media/jian/ssd4t/zero/zero/v4/config/expbase_voxel_size_0.005.yaml
# python /data/zero/zero/v4/trainer_lotus.py --config $config_path --name $name
