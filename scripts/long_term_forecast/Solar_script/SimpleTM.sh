export CUDA_VISIBLE_DEVICES=0

model_name=SimpleTM

python3 -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ../dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --use_norm 0 \
  --wv "db8" \
  --m 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  --clusterGNN True \
  --num_clusters 40


python3 -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ../dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.003 \
  --batch_size 256 \
  --use_norm 0 \
  --wv "db8" \
  --m 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --clusterGNN True \
  --use_norm 0 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  --num_clusters 2


python3 -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ../dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.003 \
  --batch_size 256 \
  --use_norm 0 \
  --wv "db8" \
  --m 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des "Exp" \
  --itr 1 \
  --clusterGNN True \
  --use_norm 0 \
  --alpha 0.1 \
  --l1_weight 0.005 \
  --num_clusters 40 

python3 -u run.py \
  --is_training 1 \
  --lradj 'TST' \
  --patience 3 \
  --root_path ../dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id Solar \
  --model "$model_name" \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 1 \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 0.009 \
  --batch_size 256 \
  --use_norm 0 \
  --wv "db8" \
  --m 1 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des "Exp" \
  --itr 1 \
  --clusterGNN True \
  --use_norm 0 \
  --alpha 0.0 \
  --l1_weight 0.005 \
  --num_clusters 40
