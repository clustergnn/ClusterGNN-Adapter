export CUDA_VISIBLE_DEVICES=2

model=TimeXer

python3 -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ../ \
  --data_path dataset/exchange_rate/exchange_rate.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --num_clusters 6 

python3 -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ../ \
  --data_path dataset/exchange_rate/exchange_rate.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --num_clusters 6

python3 -u run.py \
  --seed 2024 \
  --phase 0 \
  --model $model \
  --root_path ../ \
  --data_path dataset/exchange_rate/exchange_rate.csv \
  --file_path checkpoints/ \
  --mode M \
  --freq h \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --c_out 8 \
  --enc_in 8 \
  --dec_in 8 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --num_clusters 6

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model $model \
  --root_path ../dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
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
  --num_clusters 6



