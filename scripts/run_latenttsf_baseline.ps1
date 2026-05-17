param(
  [string]$Dataset = "ETTh1",
  [string]$Model = "DLinear",
  [int]$PredLen = 96,
  [string]$Python = "D:\ProgramData\Anaconda3\envs\latentworld\python.exe",
  [string]$Device = "cuda"
)

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT
$env:WANDB_MODE = "offline"

$datasetConfigs = @{
  "ETTh1" = @{
    data = "ETTh1"; root = ".\dataset\ETT-small\"; file = "ETTh1.csv"; freq = "h";
    enc_in = 7; dec_in = 7; c_out = 7; d_model = 32; d_ff = 64;
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "ETTh2" = @{
    data = "ETTh2"; root = ".\dataset\ETT-small\"; file = "ETTh2.csv"; freq = "h";
    enc_in = 7; dec_in = 7; c_out = 7; d_model = 64; d_ff = 128;
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTh2_AE_ETTh2_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "ETTm1" = @{
    data = "ETTm1"; root = ".\dataset\ETT-small\"; file = "ETTm1.csv"; freq = "t";
    enc_in = 7; dec_in = 7; c_out = 7; d_model = 32; d_ff = 64;
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTm1_AE_ETTm1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "ETTm2" = @{
    data = "ETTm2"; root = ".\dataset\ETT-small\"; file = "ETTm2.csv"; freq = "t";
    enc_in = 7; dec_in = 7; c_out = 7; d_model = 64; d_ff = 128;
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_ETTm2_AE_ETTm2_ftM_sl24_dm64_dff128_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
  "exchange_rate" = @{
    data = "custom"; root = ".\dataset\exchange_rate\"; file = "exchange_rate.csv"; freq = "d";
    enc_in = 8; dec_in = 8; c_out = 8; d_model = 128; d_ff = 256;
    ae = ".\checkpoints\AutoEncoder_MLP_MAE_exchange_rate_AE_custom_ftM_sl24_dm128_dff256_lradj0_Exp-sl24-lr0.0005-500-32bs_0\checkpoint.pth"
  }
}

$modelConfigs = @{
  "DLinear" = @{
    lr = "0.001"; extra = @("--moving_avg", "25")
  }
  "iTransformer" = @{
    lr = "0.001"; extra = @("--n_heads", "4", "--e_layers", "2", "--factor", "1", "--activation", "gelu", "--dropout", "0.1")
  }
  "PatchTST" = @{
    lr = "0.0005"; extra = @("--n_heads", "4", "--e_layers", "2", "--factor", "1", "--activation", "gelu", "--dropout", "0.1", "--patch_len", "16")
  }
  "TimesNet" = @{
    lr = "0.001"; extra = @("--e_layers", "2", "--top_k", "5", "--num_kernels", "6", "--dropout", "0.1")
  }
  "TimeMixer" = @{
    lr = "0.001"; extra = @("--e_layers", "2", "--dropout", "0.1", "--channel_independence", "1", "--decomp_method", "moving_avg", "--use_norm", "1", "--down_sampling_layers", "0", "--down_sampling_window", "1")
  }
}

if (-not $datasetConfigs.ContainsKey($Dataset)) {
  throw "Unsupported dataset: $Dataset. Choices: $($datasetConfigs.Keys -join ', ')"
}
if (-not $modelConfigs.ContainsKey($Model)) {
  throw "Unsupported model: $Model. Choices: $($modelConfigs.Keys -join ', ')"
}

$dc = $datasetConfigs[$Dataset]
$mc = $modelConfigs[$Model]
$outputDir = ".\checkpoints\latenttsf_${Model}_${Dataset}_sl96_pl${PredLen}"

$cmd = @(
  "-u", ".\train_latent_proto_regularized.py",
  "--output_dir", $outputDir,
  "--autoencoder_path", $dc.ae,
  "--model", $Model,
  "--task_name", "long_term_forecast",
  "--data", $dc.data,
  "--root_path", $dc.root,
  "--data_path", $dc.file,
  "--features", "M",
  "--target", "OT",
  "--freq", $dc.freq,
  "--seq_len", "96",
  "--label_len", "0",
  "--pred_len", "$PredLen",
  "--step", "1",
  "--enc_in", "$($dc.enc_in)",
  "--dec_in", "$($dc.dec_in)",
  "--c_out", "$($dc.c_out)",
  "--d_model", "$($dc.d_model)",
  "--d_ff", "$($dc.d_ff)",
  "--ae_type", "MLP",
  "--lambda_latent", "0",
  "--lambda_delta", "0",
  "--lambda_proto", "0",
  "--epochs", "40",
  "--patience", "8",
  "--batch_size", "128",
  "--num_workers", "0",
  "--lr", $mc.lr,
  "--weight_decay", "0.0001",
  "--grad_clip", "1.0",
  "--device", $Device
)
$cmd += $mc.extra

Write-Host "Running baseline: model=$Model dataset=$Dataset pred_len=$PredLen"
Write-Host "Output dir: $outputDir"
& $Python $cmd
