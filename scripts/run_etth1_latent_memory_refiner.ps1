param(
  [string]$Model = "DLinear",
  [int]$PredLen = 96,
  [int]$K = 16,
  [double]$Alpha = 0.02,
  [string]$GateMode = "fixed_late",
  [int]$Hidden = 256,
  [double]$LambdaAnchor = 0.5,
  [double]$LambdaDiv = 0,
  [double]$LambdaSlotNorm = 0.005,
  [double]$LambdaUsage = 0.0001,
  [double]$LR = 0.0001,
  [int]$Epochs = 20,
  [int]$Patience = 5,
  [int]$BatchSize = 32,
  [string]$OutputDir = "",
  [string]$Python = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe",
  [string]$Device = "cuda"
)

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$dataset = @{
  data = "ETTh1"
  root = ".\dataset\ETT-small\"
  file = "ETTh1.csv"
  freq = "h"
  enc_in = 7
  dec_in = 7
  c_out = 7
  d_model = 32
  d_ff = 64
  ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
}

$baseCheckpoint = ".\checkpoints\latenttsf_${Model}_ETTh1_sl96_pl${PredLen}\best_latent_proto_regularized.pt"
if (-not (Test-Path $baseCheckpoint)) {
  throw "Missing base checkpoint: $baseCheckpoint"
}

if ($OutputDir -eq "") {
  $alphaTag = ("{0:0.###}" -f $Alpha).Replace(".", "")
  $gateTag = $GateMode.Replace("_", "")
  $usageTag = ("{0:0.####}" -f $LambdaUsage).Replace(".", "")
  $slotNormTag = ("{0:0.####}" -f $LambdaSlotNorm).Replace(".", "")
  $anchorTag = ("{0:0.###}" -f $LambdaAnchor).Replace(".", "")

  $OutputDir = ".\checkpoints\latent_memory_refiner_${Model}_ETTh1_sl96_pl${PredLen}_K${K}_a${alphaTag}_${gateTag}_anc${anchorTag}_u${usageTag}_sn${slotNormTag}"
}

Write-Host "Running latent memory refiner: model=$Model pred_len=$PredLen gate=$GateMode K=$K alpha=$Alpha"
Write-Host "Base checkpoint: $baseCheckpoint"
Write-Host "Output dir: $OutputDir"

& $Python -u .\train_latent_memory_refiner.py `
  --data $dataset.data `
  --root_path $dataset.root `
  --data_path $dataset.file `
  --features M `
  --target OT `
  --freq $dataset.freq `
  --seq_len 96 `
  --label_len 0 `
  --pred_len $PredLen `
  --step 1 `
  --enc_in $dataset.enc_in `
  --dec_in $dataset.dec_in `
  --c_out $dataset.c_out `
  --d_model $dataset.d_model `
  --d_ff $dataset.d_ff `
  --ae_type MLP `
  --model $Model `
  --autoencoder_path $dataset.ae `
  --base_checkpoint $baseCheckpoint `
  --K $K `
  --hidden $Hidden `
  --alpha $Alpha `
  --gate_mode $GateMode `
  --query_type temporal_conv `
  --lambda_anchor $LambdaAnchor `
  --lambda_div $LambdaDiv `
  --lambda_slot_norm $LambdaSlotNorm `
  --lambda_usage $LambdaUsage `
  --epochs $Epochs `
  --patience $Patience `
  --batch_size $BatchSize `
  --num_workers 0 `
  --lr $LR `
  --weight_decay 1e-4 `
  --grad_clip 1.0 `
  --device $Device `
  --output_dir $OutputDir
