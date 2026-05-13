$env:WANDB_MODE = "offline"

$AE_PATH = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
$MB_PATH = ".\latent_outputs\ETTh1_sl96_pl96_dm32_dff64_MLP\ablation_norm_prior\best_multibranch.pt"
$OUT_DIR = ".\latent_outputs\ETTh1_sl96_pl96_dm32_dff64_MLP\decoded_eval_norm_prior"

python -u .\evaluate_multibranch_decoded.py `
  --multibranch_path $MB_PATH `
  --autoencoder_path $AE_PATH `
  --output_dir $OUT_DIR `
  --eval_flags val,test `
  --save_predictions `
  --task_name long_term_forecast `
  --data ETTh1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --features M `
  --seq_len 96 `
  --label_len 0 `
  --pred_len 96 `
  --step 1 `
  --enc_in 7 `
  --d_model 32 `
  --d_ff 64 `
  --ae_type MLP `
  --batch_size 128 `
  --num_workers 0 `
  --device cuda
