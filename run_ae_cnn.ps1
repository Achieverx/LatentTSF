$env:CUDA_VISIBLE_DEVICES = "0"
$env:WANDB_MODE = "offline"

D:\ProgramData\Anaconda3\envs\latentworld\python.exe -u .\my_AE.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --model_id ETTh1_AE_CNN `
  --model DLinear `
  --data ETTh1 `
  --features M `
  --seq_len 24 `
  --label_len 0 `
  --pred_len 96 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 32 `
  --d_ff 64 `
  --train_epochs 500 `
  --batch_size 32 `
  --learning_rate 0.0005 `
  --patience 20 `
  --seed 42 `
  --use_lradj 0 `
  --ae_type CNN `
  --ae_loss MAE `
  --des Exp-CNN-sl24-lr0.0005-500-32bs `
  --itr 1