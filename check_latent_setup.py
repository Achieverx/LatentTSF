import argparse
import os

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
import torch
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder


def load_autoencoder(args, device):
    autoencoder = get_autoencoder(args).float().to(device)
    state_dict = torch.load(args.autoencoder_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "autoencoder_state_dict" in state_dict:
        state_dict = state_dict["autoencoder_state_dict"]
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return autoencoder


def check_csv(args):
    path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, nrows=5)
    expected_cols = args.enc_in + 1
    if len(df.columns) != expected_cols:
        raise ValueError(
            f"CSV column count mismatch: got {len(df.columns)} columns, expected date + enc_in = {expected_cols}."
        )
    if "date" not in df.columns:
        raise ValueError("CSV must contain a date column.")
    print(f"CSV OK: {path}")
    print(f"  rows(sampled)=5 columns={len(df.columns)} enc_in={args.enc_in}")
    print(f"  first date={df['date'].iloc[0]}")


def check_loaders(args):
    for flag in ["train", "val", "test"]:
        dataset, loader = data_provider(args, flag)
        print(f"{flag} loader OK: samples={len(dataset)} batches={len(loader)}")
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
        print(f"  x={tuple(batch_x.shape)} y={tuple(batch_y.shape)}")
        if batch_x.shape[-1] != args.enc_in or batch_y.shape[-1] != args.enc_in:
            raise ValueError(f"{flag} feature dim mismatch.")
        if batch_x.shape[1] != args.seq_len:
            raise ValueError(f"{flag} seq_len mismatch.")
        if batch_y.shape[1] != args.label_len + args.pred_len:
            raise ValueError(f"{flag} y length mismatch.")


def check_autoencoder(args, device):
    autoencoder = load_autoencoder(args, device)
    _, loader = data_provider(args, "val")
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(loader))
    y = batch_y[:, -args.pred_len :, :].float().to(device)
    with torch.no_grad():
        z_y = autoencoder.encode(y)
        y_recon = autoencoder.decode(z_y)
    mse = F.mse_loss(y_recon, y).item()
    mae = (y_recon - y).abs().mean().item()
    print("AE OK:")
    print(f"  z_y={tuple(z_y.shape)} recon={tuple(y_recon.shape)}")
    print(f"  val first-batch recon MSE/MAE={mse:.6f}/{mae:.6f}")
    if z_y.shape[-1] != args.d_model:
        raise ValueError(f"AE d_model mismatch: got {z_y.shape[-1]}, expected {args.d_model}.")
    if y_recon.shape != y.shape:
        raise ValueError("AE decode shape mismatch.")


def main():
    parser = argparse.ArgumentParser(description="Minimal sanity check for latent forecasting setup")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--data", type=str, default="ETTh2")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh2.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=336)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    check_csv(args)
    check_loaders(args)
    check_autoencoder(args, device)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
