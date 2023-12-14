export CUDA_VISIBLE_DEVICES=1
python main.py \
--n_class 5 \
--data_path "./graphs/simclr_files/" \
--val_set "./data/val_set.txt" \
--model_path "./graph_transformer/saved_models/" \
--log_path "./graph_transformer/runs/" \
--task_name "GraphCAM" \
--batch_size 1 \
--test \
--log_interval_local 6 \
--resume "./graph_transformer/saved_models/GraphCAM_WBCE_BS4.pth" \
--graphcam
