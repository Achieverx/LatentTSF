param(
  [string]$Python = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe",
  [string]$Device = "cuda",
  [int]$Epochs = 20,
  [int]$Patience = 5,
  [int]$BatchSize = 32
)

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

$logsDir = ".\logs"
$resultsDir = ".\results"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$outLog = Join-Path $logsDir "gated_lora_calibration_sweep.out.log"
$errLog = Join-Path $logsDir "gated_lora_calibration_sweep.err.log"
if (Test-Path $outLog) { Remove-Item $outLog -Force }
if (Test-Path $errLog) { Remove-Item $errLog -Force }

$runner = Join-Path $PSScriptRoot "run_etth1_lora_lowrank_latent_adaptation.ps1"

function Invoke-Experiment {
  param(
    [string]$Model,
    [int]$PredLen,
    [int]$UseLora,
    [string]$LoraTarget,
    [string]$CalibType,
    [string]$CalibGate,
    [string]$Tag,
    [string]$OutputTag,
    [double]$LambdaCalBaseAnchor = 0.0
  )

  $outputDir = ".\checkpoints\gated_lora_calibration_${Model}_ETTh1_sl96_pl${PredLen}_${OutputTag}"
  Add-Content -Path $outLog -Value ("[{0}] START {1} model={2} pred_len={3} use_lora={4} lora_target={5} calib={6} calib_gate={7}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Tag, $Model, $PredLen, $UseLora, $LoraTarget, $CalibType, $CalibGate)
  & $runner `
    -Model $Model `
    -PredLen $PredLen `
    -UseLora $UseLora `
    -LoraRank 4 `
    -LoraAlpha 8 `
    -LoraTarget $LoraTarget `
    -CalibType $CalibType `
    -CalibRank 4 `
    -CalibGate $CalibGate `
    -DynamicScale none `
    -Alpha 0.02 `
    -LambdaLoraAnchor 0.1 `
    -LambdaCalAnchor 0.5 `
    -LambdaCalBaseAnchor $LambdaCalBaseAnchor `
    -LambdaNorm 0.005 `
    -LambdaSmooth 0.01 `
    -LambdaGate 0.01 `
    -GateInit 0.5 `
    -GateRange 1.0 `
    -GateHidden 64 `
    -LR 1e-4 `
    -Epochs $Epochs `
    -Patience $Patience `
    -BatchSize $BatchSize `
    -Device $Device `
    -Python $Python `
    -OutputDir $outputDir 1>> $outLog 2>> $errLog

  if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed: $Tag model=$Model pred_len=$PredLen"
  }

  Add-Content -Path $outLog -Value ("[{0}] END   {1} model={2} pred_len={3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Tag, $Model, $PredLen)
}

$models = @("DLinear", "PatchTST", "iTransformer")
$predLens = @(96, 192)

foreach ($model in $models) {
  foreach ($predLen in $predLens) {
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 0 -LoraTarget auto -CalibType none -CalibGate none -Tag baseline -OutputTag "baseline"
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 0 -LoraTarget auto -CalibType lowrank -CalibGate none -Tag lowrank_only -OutputTag "lowrank"
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 1 -LoraTarget auto -CalibType none -CalibGate none -Tag lora_only -OutputTag "lora"
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 1 -LoraTarget auto -CalibType lowrank -CalibGate none -Tag old_lora_lowrank -OutputTag "old_lora_lowrank"
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 1 -LoraTarget auto -CalibType lowrank -CalibGate scalar -Tag gated_lora_lowrank -OutputTag "gated_lora_lowrank" -LambdaCalBaseAnchor 0.1
  }
}

Invoke-Experiment -Model DLinear -PredLen 96 -UseLora 1 -LoraTarget auto -CalibType block -CalibGate block -Tag gated_lora_dynamicblock -OutputTag "gated_lora_dynamicblock" -LambdaCalBaseAnchor 0.1
Invoke-Experiment -Model iTransformer -PredLen 192 -UseLora 1 -LoraTarget qv_only -CalibType none -CalibGate none -Tag qv_only_lora -OutputTag "qv_only_lora"
Invoke-Experiment -Model iTransformer -PredLen 192 -UseLora 1 -LoraTarget qv_only -CalibType lowrank -CalibGate scalar -Tag qv_only_gated_lora_lowrank -OutputTag "qv_only_gated_lora_lowrank" -LambdaCalBaseAnchor 0.1

& $Python -u .\summarize_gated_lora_calibration_results.py `
  --root_dir .\checkpoints `
  --dir_prefix gated_lora_calibration_ `
  --output_csv .\results\gated_lora_calibration_etth1_summary.csv 1>> $outLog 2>> $errLog

if ($LASTEXITCODE -ne 0) {
  throw "Summary generation failed with exit code $LASTEXITCODE"
}

Add-Content -Path $outLog -Value ("[{0}] Sweep complete" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
