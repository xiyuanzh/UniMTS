for k in 1 2 3 5 10
do

python finetune.py \
--mode full \
--k $k \
--batch_size 64 \
--num_epochs 200 \
--checkpoint './checkpoint/UniMTS.pth' \
--data_path 'UniMTS_data'

done

python finetune.py \
--mode full \
--batch_size 64 \
--num_epochs 200 \
--checkpoint './checkpoint/UniMTS.pth' \
--data_path 'UniMTS_data'
