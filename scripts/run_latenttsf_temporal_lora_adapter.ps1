$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$AE_PATH = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
$LATENTTSF_CKPT = ".\checkpoints\latenttsf_pure_ETTh1_sl96_pl96\best_latent_proto_regularized.pt"
$PYTHON = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe"

foreach ($RT in @("4", "8")) {
  & $PYTHON -u .\train_latenttsf_temporal_lora_adapter.py `
    --output_dir ".\latent_outputs\temporal_lora_purebase_ETTh1_sl96_pl96_rt${RT}_rc2_alpha0003" `
    --autoencoder_path $AE_PATH `
    --latenttsf_checkpoint $LATENTTSF_CKPT `
    --model DLinear `
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
    --rank_time $RT `
    --rank_channel 2 `
    --hidden_dim 64 `
    --alpha_init 0.003 `
    --alpha_max 0.2 `
    --lambda_delta 0.001 `
    --epochs 40 `
    --patience 8 `
    --batch_size 128 `
    --num_workers 0 `
    --lr 0.0001 `
    --weight_decay 0.0001 `
    --grad_clip 1.0 `
    --device cuda
}
