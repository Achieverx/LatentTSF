param(
  [string]$Python = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe",
  [string]$Device = "cuda"
)

$models = @("DLinear", "iTransformer", "PatchTST", "TimesNet", "TimeMixer")

foreach ($model in $models) {
  powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\run_latenttsf_baseline.ps1" -Dataset "exchange_rate" -Model $model -Python $Python -Device $Device
  powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\run_channel_adapter.ps1" -Dataset "exchange_rate" -Model $model -Python $Python -Device $Device
}
