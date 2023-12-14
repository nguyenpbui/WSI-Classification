export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 5 \
--data_path "./graphs/simclr_files/" \
--train_set "./data/train_set_fold_3.txt" \
--val_set "./data/valid_set_fold_3.txt" \
--model_path "./graph_transformer/saved_models/" \
--log_path "./graph_transformer/runs/" \
--task_name "GraphCAM_F3" \
--batch_size 4 \
--train \
# --graphcam \
# --log_interval_local 12 \
