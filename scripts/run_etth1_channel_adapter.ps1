$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$PYTHON = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe"
$AE_PATH = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
$BASELINE_DIR = ".\checkpoints\latenttsf_pure_ETTh1_sl96_pl96"
$BASELINE_CKPT = Join-Path $BASELINE_DIR "best_latent_proto_regularized.pt"

& $PYTHON -u .\train_latenttsf_channel_adapter.py `
  --output_dir ".\checkpoints\latenttsf_channel_adapter_ETTh1_sl96_pl96" `
  --autoencoder_path $AE_PATH `
  --latenttsf_checkpoint $BASELINE_CKPT `
  --model iTransformer `
  --task_name long_term_forecast `
  --data ETTh1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --features M `
  --target OT `
  --freq h `
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
  --adapter_hidden_dim 128 `
  --dropout 0.1 `
  --alpha 0.01 `
  --lambda_reg 0.1 `
  --epochs 40 `
  --patience 8 `
  --batch_size 128 `
  --num_workers 0 `
  --lr 0.00005 `
  --weight_decay 0.0001 `
  --grad_clip 1.0 `
  --device cuda
