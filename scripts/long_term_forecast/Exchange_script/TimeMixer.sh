export CUDA_VISIBLE_DEVICES=0
model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --clusterGNN True \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --num_clusters 6


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --clusterGNN True \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --num_clusters 6


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --clusterGNN True \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --num_clusters 6

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --clusterGNN True \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 32 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --num_clusters 6

