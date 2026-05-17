param(
  [string]$Dataset = "ETTh1",
  [string]$Model = "DLinear",
  [int]$PredLen = 96,
  [string]$AdapterMode = "full",
  [string]$Python = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe",
  [string]$Device = "cuda"
)

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$datasetConfigs = @{
  "ETTh1" = @{
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "ETTh2" = @{
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh2_AE_ETTh2_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "ETTm1" = @{
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTm1_AE_ETTm1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "ETTm2" = @{
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTm2_AE_ETTm2_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "exchange_rate" = @{
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_exchange_rate_AE_custom_ftM_sl24_dm128_dff256_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
}

if (-not $datasetConfigs.ContainsKey($Dataset)) {
  throw "Unsupported dataset: $Dataset. Choices: $($datasetConfigs.Keys -join ', ')"
}

$dc = $datasetConfigs[$Dataset]
$baselineDir = ".\checkpoints\latenttsf_${Model}_${Dataset}_sl96_pl${PredLen}"
$baselineCkpt = Join-Path $baselineDir "best_latent_proto_regularized.pt"
$outputDir = ".\checkpoints\corr_mismatch_adapter_${AdapterMode}_${Model}_${Dataset}_sl96_pl${PredLen}"

if (-not (Test-Path $baselineCkpt)) {
  throw "Baseline checkpoint not found: $baselineCkpt . Run scripts/run_latenttsf_baseline.ps1 first."
}

Write-Host "Running Corr-Mismatch Adapter: mode=$AdapterMode model=$Model dataset=$Dataset pred_len=$PredLen"
Write-Host "Baseline ckpt: $baselineCkpt"

& $Python -u .\train_latenttsf_corr_mismatch_adapter.py `
  --output_dir $outputDir `
  --autoencoder_path $dc.ae `
  --latenttsf_checkpoint $baselineCkpt `
  --adapter_mode $AdapterMode `
  --adapter_hidden_dim 128 `
  --dropout 0.1 `
  --alpha 0.01 `
  --lambda_reg 0.1 `
  --lambda_corr 0.01 `
  --epochs 40 `
  --patience 8 `
  --batch_size 128 `
  --num_workers 0 `
  --lr 0.00005 `
  --weight_decay 0.0001 `
  --grad_clip 1.0 `
  --device $Device
