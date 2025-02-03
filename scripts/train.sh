
conda activate zero

name=expBins_first_try
config_path=/media/jian/ssd4t/zero/zero/expBins/config/expbase_voxel_size_0.005_bins.yaml

python /media/jian/ssd4t/zero/zero/expBins/trainer_expbase.py --config $config_path --name $name


