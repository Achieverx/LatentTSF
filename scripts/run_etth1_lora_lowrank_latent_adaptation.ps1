param(
  [string]$Model = "DLinear",
  [int]$PredLen = 96,
  [int]$UseLora = 1,
  [int]$LoraRank = 4,
  [double]$LoraAlpha = 8,
  [string]$LoraTarget = "auto",
  [string]$CalibType = "lowrank",
  [int]$CalibRank = 4,
  [string]$CalibGate = "scalar",
  [string]$DynamicScale = "none",
  [double]$Alpha = 0.02,
  [double]$LambdaLoraAnchor = 0.1,
  [double]$LambdaCalAnchor = 0.5,
  [double]$LambdaCalBaseAnchor = 0.1,
  [double]$LambdaNorm = 0.005,
  [double]$LambdaSmooth = 0.01,
  [double]$LambdaGate = 0.01,
  [double]$GateInit = 0.5,
  [double]$GateRange = 1.0,
  [int]$GateHidden = 64,
  [double]$LR = 1e-4,
  [int]$Epochs = 20,
  [int]$Patience = 5,
  [int]$BatchSize = 32,
  [string]$Device = "cuda",
  [string]$Python = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe",
  [string]$OutputDir = ""
)

function Format-Tag([double]$Value) {
  return ("{0:0.###}" -f $Value).Replace(".", "p")
}

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$dataset = @{
  data = "ETTh1"
  root = ".\dataset\ETT-small\"
  file = "ETTh1.csv"
  freq = "h"
  ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
}

$baseCheckpoint = ".\checkpoints\latenttsf_${Model}_ETTh1_sl96_pl${PredLen}\best_latent_proto_regularized.pt"
if (-not (Test-Path $dataset.ae)) {
  throw "Missing autoencoder checkpoint: $($dataset.ae)"
}
if (-not (Test-Path $baseCheckpoint)) {
  throw "Missing base checkpoint: $baseCheckpoint"
}

if ($OutputDir -eq "") {
  $alphaTag = Format-Tag $Alpha
  $safeTarget = $LoraTarget.Replace("_", "")
  $OutputDir = ".\checkpoints\gated_lora_calibration_${Model}_ETTh1_sl96_pl${PredLen}_ul${UseLora}_lr${LoraRank}_lt${safeTarget}_${CalibType}_cr${CalibRank}_cg${CalibGate}_a${alphaTag}"
}

Write-Output "Running backbone-aware gated latent adaptation: model=$Model pred_len=$PredLen use_lora=$UseLora calib=$CalibType calib_gate=$CalibGate"
Write-Output "Autoencoder checkpoint: $($dataset.ae)"
Write-Output "Base checkpoint: $baseCheckpoint"
Write-Output "Output dir: $OutputDir"

& $Python -u .\train_lora_lowrank_latent_adaptation.py `
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
  --ae_type MLP `
  --model $Model `
  --autoencoder_path $dataset.ae `
  --base_checkpoint $baseCheckpoint `
  --output_dir $OutputDir `
  --device $Device `
  --use_lora $UseLora `
  --lora_rank $LoraRank `
  --lora_alpha $LoraAlpha `
  --lora_dropout 0.0 `
  --lora_target $LoraTarget `
  --calib_type $CalibType `
  --calib_rank $CalibRank `
  --alpha $Alpha `
  --calib_gate $CalibGate `
  --dynamic_scale $DynamicScale `
  --gate_hidden $GateHidden `
  --gate_range $GateRange `
  --gate_init $GateInit `
  --lambda_lora_anchor $LambdaLoraAnchor `
  --lambda_cal_anchor $LambdaCalAnchor `
  --lambda_cal_base_anchor $LambdaCalBaseAnchor `
  --lambda_norm $LambdaNorm `
  --lambda_smooth $LambdaSmooth `
  --lambda_gate $LambdaGate `
  --epochs $Epochs `
  --patience $Patience `
  --batch_size $BatchSize `
  --num_workers 0 `
  --lr $LR `
  --weight_decay 1e-4 `
  --grad_clip 1.0

if ($LASTEXITCODE -ne 0) {
  throw "Training failed with exit code $LASTEXITCODE"
}
