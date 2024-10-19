python evaluate_custom.py \
--batch_size 64 \
--checkpoint './checkpoint/UniMTS.pth' \
--X_path 'UniMTS_data/TNDA-HAR/X_test.npy' \
--y_path 'UniMTS_data/TNDA-HAR/y_test.npy' \
--config_path 'UniMTS_data/TNDA-HAR/TNDA-HAR.json' \
--joint_list 20 2 21 3 11 \
--original_sampling_rate 50 