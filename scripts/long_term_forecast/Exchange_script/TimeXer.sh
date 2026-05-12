export CUDA_VISIBLE_DEVICES=0

model=TimeXer

python3 -u run.py \
  --is_training 1 \
  --model $model \
  --model_id Exchange_96_96 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --clusterGNN True \
  --num_clusters 6 

python3 -u run.py \
  --is_training 1 \
  --model $model \
  --model_id Exchange_96_192 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --clusterGNN True \
  --num_clusters 6

python3 -u run.py \
  --is_training 1 \
  --model $model \
  --model_id Exchange_96_336 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --clusterGNN True \
  --num_clusters 6

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model $model \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --clusterGNN True \
  --num_clusters 6



