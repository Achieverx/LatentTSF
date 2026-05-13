$env:WANDB_MODE = "offline"

$SOURCE_DIR = ".\latent_outputs\official_multibranch_ETTh1_sl96_pl96_normK8"
$OUT_DIR = ".\latent_outputs\official_multibranch_ETTh1_sl96_pl96_normK8_sparse_moe_reg"

# Closest-to-base residual gate setting:
# test MSE ~= 0.41046, essentially falling back to z_base.
D:\ProgramData\Anaconda3\envs\latentworld\python.exe -u .\train_sparse_moe_gate.py `
  --source_dir $SOURCE_DIR `
  --output_dir $OUT_DIR `
  --top_m 0 `
  --moe_tau 1.0 `
  --alpha_init_bias -3.0 `
  --alpha_reg 0.1 `
  --res_reg 0.0 `
  --hidden_dim 256 `
  --epochs 50 `
  --patience 8 `
  --batch_size 128 `
  --lr 0.001 `
  --weight_decay 0.0001 `
  --device cuda
