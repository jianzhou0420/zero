
conda activate zero

name=verify_expbasev4
config_path=/media/jian/ssd4t/zero/zero/expbasev4/config/expbase_voxel_size_0.005.yaml

python /media/jian/ssd4t/zero/zero/expbasev4/trainer_expbase.py --config $config_path --name $name


