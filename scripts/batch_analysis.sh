

# 先生成数据再分析
conda activate zero
batch_file='/media/jian/ssd4t/zero/zero/v3/config/0.005_sort_shape.pickle'

python /media/jian/ssd4t/zero/zero/v3/dataset/dataset_v5.py \
    --config /media/jian/ssd4t/zero/zero/v3/config/0.005_sort_shape.yaml \
    --output $batch_file

python /media/jian/ssd4t/zero/zero/v3/dataprocess/batch_analysis.py  \
    --batch_file $batch_file


rm  /media/jian/ssd4t/zero/zero/v3/config/0.005_sort_shape.pickle