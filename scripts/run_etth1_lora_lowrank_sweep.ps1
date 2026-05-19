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

$outLog = Join-Path $logsDir "lora_lowrank_latent_adaptation_sweep.out.log"
$errLog = Join-Path $logsDir "lora_lowrank_latent_adaptation_sweep.err.log"
if (Test-Path $outLog) { Remove-Item $outLog -Force }
if (Test-Path $errLog) { Remove-Item $errLog -Force }

$runner = Join-Path $PSScriptRoot "run_etth1_lora_lowrank_latent_adaptation.ps1"

function Invoke-Experiment {
  param(
    [string]$Model,
    [int]$PredLen,
    [int]$UseLora,
    [string]$CalibType,
    [int]$CalibRank,
    [string]$DynamicScale,
    [string]$Tag
  )

  Add-Content -Path $outLog -Value ("[{0}] START {1} model={2} pred_len={3} use_lora={4} calib={5} dynamic_scale={6}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Tag, $Model, $PredLen, $UseLora, $CalibType, $DynamicScale)
  & $runner `
    -Model $Model `
    -PredLen $PredLen `
    -UseLora $UseLora `
    -LoraRank 4 `
    -LoraAlpha 8 `
    -LoraTarget auto `
    -CalibType $CalibType `
    -CalibRank $CalibRank `
    -DynamicScale $DynamicScale `
    -Alpha 0.02 `
    -LambdaLoraAnchor 0.1 `
    -LambdaCalAnchor 0.5 `
    -LambdaNorm 0.005 `
    -LambdaSmooth 0.01 `
    -LambdaScale 0.01 `
    -LR 1e-4 `
    -Epochs $Epochs `
    -Patience $Patience `
    -BatchSize $BatchSize `
    -Device $Device `
    -Python $Python 1>> $outLog 2>> $errLog

  if ($LASTEXITCODE -ne 0) {
    throw "Experiment failed: $Tag model=$Model pred_len=$PredLen"
  }

  Add-Content -Path $outLog -Value ("[{0}] END   {1} model={2} pred_len={3}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Tag, $Model, $PredLen)
}

$models = @("DLinear", "PatchTST", "iTransformer")
$predLens = @(96, 192)

foreach ($model in $models) {
  foreach ($predLen in $predLens) {
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 0 -CalibType none -CalibRank 4 -DynamicScale none -Tag baseline
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 0 -CalibType lowrank -CalibRank 4 -DynamicScale none -Tag lowrank_only
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 1 -CalibType none -CalibRank 4 -DynamicScale none -Tag lora_only
    Invoke-Experiment -Model $model -PredLen $predLen -UseLora 1 -CalibType lowrank -CalibRank 4 -DynamicScale none -Tag lora_lowrank
  }
}

Invoke-Experiment -Model DLinear -PredLen 96 -UseLora 0 -CalibType block -CalibRank 4 -DynamicScale none -Tag static_block
Invoke-Experiment -Model DLinear -PredLen 96 -UseLora 0 -CalibType block -CalibRank 4 -DynamicScale block -Tag dynamic_block
Invoke-Experiment -Model DLinear -PredLen 96 -UseLora 1 -CalibType block -CalibRank 4 -DynamicScale block -Tag lora_dynamic_block

& $Python -u .\summarize_lora_lowrank_results.py `
  --root_dir .\checkpoints `
  --output_csv .\results\lora_lowrank_etth1_summary.csv 1>> $outLog 2>> $errLog

if ($LASTEXITCODE -ne 0) {
  throw "Summary generation failed with exit code $LASTEXITCODE"
}

Add-Content -Path $outLog -Value ("[{0}] Sweep complete" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
