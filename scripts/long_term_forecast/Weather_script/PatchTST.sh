export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3 \
  --n_heads 4  \
  --num_clusters 7

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3 \
  --n_heads 16  \
  --num_clusters 7

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3 \
  --n_heads 4 \
  --batch_size 128 \
  --num_clusters 18

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3 \
  --n_heads 4 \
  --batch_size 128 \
  --num_clusters 13