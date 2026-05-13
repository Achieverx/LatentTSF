$env:WANDB_MODE = "offline"

$AE_PATH = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
$OUT_DIR = ".\latent_outputs\official_multibranch_ETTh1_sl96_pl96_normK8"

python -u .\official_multibranch_pipeline.py `
  --output_dir $OUT_DIR `
  --autoencoder_path $AE_PATH `
  --task_name long_term_forecast `
  --model DLinear `
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
  --num_branches 8 `
  --hidden_dim 512 `
  --epochs 80 `
  --patience 10 `
  --batch_size 128 `
  --lr 0.001 `
  --best_weight 0.1 `
  --fuse_weight 0.3 `
  --score_weight 0.1 `
  --score_tau 0.05 `
  --early_stop_metric assign_mse `
  --num_workers 0 `
  --device cuda `
  --save_predictions
