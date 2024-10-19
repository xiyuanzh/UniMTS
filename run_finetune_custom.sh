for k in 1 2 3 5 10
do

python finetune_custom.py \
--mode full \
--k $k \
--batch_size 64 \
--num_epochs 200 \
--checkpoint './checkpoint/UniMTS.pth' \
--X_train_path 'UniMTS_data/TNDA-HAR/X_train.npy' \
--y_train_path 'UniMTS_data/TNDA-HAR/y_train.npy' \
--X_test_path 'UniMTS_data/TNDA-HAR/X_test.npy' \
--y_test_path 'UniMTS_data/TNDA-HAR/y_test.npy' \
--config_path 'UniMTS_data/TNDA-HAR/TNDA-HAR.json' \
--joint_list 20 2 21 3 11 \
--original_sampling_rate 50 \
--num_class 8

done

python finetune_custom.py \
--mode full \
--batch_size 64 \
--num_epochs 200 \
--checkpoint './checkpoint/UniMTS.pth' \
--X_train_path 'UniMTS_data/TNDA-HAR/X_train.npy' \
--y_train_path 'UniMTS_data/TNDA-HAR/y_train.npy' \
--X_test_path 'UniMTS_data/TNDA-HAR/X_test.npy' \
--y_test_path 'UniMTS_data/TNDA-HAR/y_test.npy' \
--config_path 'UniMTS_data/TNDA-HAR/TNDA-HAR.json' \
--joint_list 20 2 21 3 11 \
--original_sampling_rate 50 \
--num_class 8
