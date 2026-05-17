$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$PYTHON = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe"
$AE_PATH = ".\checkpoints\AutoEncoder_MLP_MAE_ETTm1_AE_ETTm1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"

& $PYTHON -u .\train_latent_proto_regularized.py `
  --output_dir ".\checkpoints\latenttsf_pure_ETTm1_sl96_pl96" `
  --autoencoder_path $AE_PATH `
  --model DLinear `
  --task_name long_term_forecast `
  --data ETTm1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTm1.csv `
  --features M `
  --target OT `
  --freq t `
  --seq_len 96 `
  --label_len 0 `
  --pred_len 96 `
  --step 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 32 `
  --d_ff 64 `
  --ae_type MLP `
  --moving_avg 25 `
  --lambda_latent 0 `
  --lambda_delta 0 `
  --lambda_proto 0 `
  --epochs 40 `
  --patience 8 `
  --batch_size 128 `
  --num_workers 0 `
  --lr 0.001 `
  --weight_decay 0.0001 `
  --grad_clip 1.0 `
  --device cuda
