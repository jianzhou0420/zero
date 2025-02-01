
conda activate zero

# name=insert_peg_0.01
# config_path=/media/jian/ssd4t/zero/zero/v3/config/insert_peg_0.01.yaml

# python /data/zero/zero/v3/trainer_lotus.py --config $config_path --name $name


name=expbase_insert_peg_0.01
config_path=/media/jian/ssd4t/zero/zero/v4/config/expbase_voxel_size_0.01.yaml

python /data/zero/zero/v4/trainer_lotus.py --config $config_path --name $name \
--resume_version_dir /media/jian/ssd4t/zero/2_Train/2025_01_31__12:12_expbase_insert_peg_0.01/version_0
