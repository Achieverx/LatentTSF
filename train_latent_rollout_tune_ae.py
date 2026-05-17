import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from my_utils import model_dict


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def official_loader(args, flag, shuffle=False):
    data_set, _ = data_provider(args, flag)
    return DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )


def load_autoencoder(args, device):
    autoencoder = get_autoencoder(args).float().to(device)
    state_dict = torch.load(args.autoencoder_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and "autoencoder_state_dict" in state_dict:
        state_dict = state_dict["autoencoder_state_dict"]

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    autoencoder.load_state_dict(state_dict)
    return autoencoder


class LatentRolloutForecaster(torch.nn.Module):
    """
    Same rollout logic as your working train_latent_rollout_forecaster.py.

    context = z_x
    each step predicts one segment
    predicted segment is appended back into context
    """

    def __init__(self, args):
        super().__init__()

        if args.pred_len % args.num_rollout_steps != 0:
            raise ValueError("pred_len must be divisible by num_rollout_steps.")

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.segment_len = args.pred_len // args.num_rollout_steps
        self.num_rollout_steps = args.num_rollout_steps

        backbone_args = SimpleNamespace(**vars(args))
        backbone_args.pred_len = self.segment_len
        backbone_args.enc_in = args.d_model
        backbone_args.dec_in = args.d_model
        backbone_args.c_out = args.d_model

        self.backbone = model_dict[args.model].Model(backbone_args).float()

    def rollout(self, z_x, teacher_z_y=None, teacher_forcing_ratio=0.0):
        context = z_x
        preds = []

        for step in range(self.num_rollout_steps):
            pred = self.backbone(context, None, None, None)
            pred = pred[:, -self.segment_len:, :]
            preds.append(pred)

            if teacher_z_y is not None and teacher_forcing_ratio > 0.0:
                use_teacher = torch.rand((), device=z_x.device) < teacher_forcing_ratio
                if use_teacher:
                    start = step * self.segment_len
                    next_piece = teacher_z_y[:, start:start + self.segment_len, :]
                else:
                    next_piece = pred
            else:
                next_piece = pred

            context = torch.cat([context[:, self.segment_len:, :], next_piece], dim=1)

        return torch.cat(preds, dim=1)

    def forward(self, z_x, teacher_z_y=None, teacher_forcing_ratio=0.0):
        return self.rollout(z_x, teacher_z_y, teacher_forcing_ratio)


def encode_batch(args, autoencoder, batch_x, batch_y, device):
    x = batch_x.float().to(device)
    y = batch_y[:, -args.pred_len:, :].float().to(device)

    z_x = autoencoder.encode(x)
    z_y = autoencoder.encode(y)

    return x, y, z_x, z_y


def compute_losses(args, autoencoder, model, batch_x, batch_y, device, teacher_forcing_ratio=0.0):
    x, y, z_x, z_y = encode_batch(args, autoencoder, batch_x, batch_y, device)

    z_pred = model(
        z_x,
        teacher_z_y=z_y if teacher_forcing_ratio > 0.0 else None,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    y_pred = autoencoder.decode(z_pred)
    y_recon = autoencoder.decode(z_y)

    loss_latent = F.mse_loss(z_pred, z_y.detach())
    loss_obs = F.mse_loss(y_pred, y)
    loss_recon = F.mse_loss(y_recon, y)
    obs_mae = (y_pred - y).abs().mean()

    total = (
        args.lambda_latent * loss_latent
        + args.lambda_obs * loss_obs
        + args.lambda_recon * loss_recon
    )

    return total, loss_latent, loss_obs, obs_mae, loss_recon


def evaluate(args, model, autoencoder, loader, device):
    model.eval()
    autoencoder.eval()

    sums = {
        "latent_mse": 0.0,
        "obs_mse": 0.0,
        "obs_mae": 0.0,
        "recon_mse": 0.0,
        "total": 0.0,
    }
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            total, loss_latent, loss_obs, obs_mae, loss_recon = compute_losses(
                args,
                autoencoder,
                model,
                batch_x,
                batch_y,
                device,
                teacher_forcing_ratio=0.0,
            )

            bsz = batch_x.size(0)
            total_count += bsz

            sums["total"] += total.item() * bsz
            sums["latent_mse"] += loss_latent.item() * bsz
            sums["obs_mse"] += loss_obs.item() * bsz
            sums["obs_mae"] += obs_mae.item() * bsz
            sums["recon_mse"] += loss_recon.item() * bsz

    return {key: value / max(total_count, 1) for key, value in sums.items()}


def configure_trainable(args, autoencoder):
    if args.tune_autoencoder:
        for p in autoencoder.parameters():
            p.requires_grad = True
    else:
        for p in autoencoder.parameters():
            p.requires_grad = False
        autoencoder.eval()


def build_optimizer(args, model, autoencoder):
    if args.tune_autoencoder:
        params = [
            {"params": model.parameters(), "lr": args.lr},
            {"params": autoencoder.parameters(), "lr": args.ae_lr},
        ]
        grad_params = list(model.parameters()) + list(autoencoder.parameters())
    else:
        params = model.parameters()
        grad_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        params,
        weight_decay=args.weight_decay,
    )
    return optimizer, grad_params


def save_checkpoint(path, args, model, autoencoder, epoch, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "autoencoder_state_dict": autoencoder.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_checkpoint(path, model, autoencoder, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    return checkpoint


def extract_latents(args, autoencoder, device):
    autoencoder.eval()
    os.makedirs(args.latent_output_dir, exist_ok=True)

    for flag in ["train", "val", "test"]:
        loader = official_loader(args, flag, shuffle=False)

        z_x_all, z_y_all, y_all = [], [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                x = batch_x.float().to(device)
                y = batch_y[:, -args.pred_len:, :].float().to(device)

                z_x = autoencoder.encode(x)
                z_y = autoencoder.encode(y)

                z_x_all.append(z_x.cpu().numpy().astype(np.float32))
                z_y_all.append(z_y.cpu().numpy().astype(np.float32))
                y_all.append(y.cpu().numpy().astype(np.float32))

        z_x = np.concatenate(z_x_all, axis=0)
        z_y = np.concatenate(z_y_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        np.save(os.path.join(args.latent_output_dir, f"{flag}_zx.npy"), z_x)
        np.save(os.path.join(args.latent_output_dir, f"{flag}_zy.npy"), z_y)
        np.save(os.path.join(args.latent_output_dir, f"{flag}_y.npy"), y)

        print(f"Saved {flag}: z_x {z_x.shape}, z_y {z_y.shape}, y {y.shape}")


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.latent_output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    model = LatentRolloutForecaster(args).to(device)

    configure_trainable(args, autoencoder)
    optimizer, grad_params = build_optimizer(args, model, autoencoder)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    best_path = os.path.join(args.output_dir, "best_latent_rollout_tune_ae.pt")
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"Training mode | tune_autoencoder={args.tune_autoencoder} "
        f"lr={args.lr} ae_lr={args.ae_lr} "
        f"lambda_latent={args.lambda_latent} "
        f"lambda_obs={args.lambda_obs} "
        f"lambda_recon={args.lambda_recon}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()

        if args.tune_autoencoder:
            autoencoder.train()
        else:
            autoencoder.eval()

        teacher_ratio = max(
            args.teacher_forcing_end,
            args.teacher_forcing_start * (1.0 - (epoch - 1) / max(args.teacher_decay_epochs, 1)),
        )

        sums = {
            "total": 0.0,
            "latent_mse": 0.0,
            "obs_mse": 0.0,
            "obs_mae": 0.0,
            "recon_mse": 0.0,
        }
        total_count = 0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            total, loss_latent, loss_obs, obs_mae, loss_recon = compute_losses(
                args,
                autoencoder,
                model,
                batch_x,
                batch_y,
                device,
                teacher_forcing_ratio=teacher_ratio,
            )

            optimizer.zero_grad()
            total.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)

            optimizer.step()

            bsz = batch_x.size(0)
            total_count += bsz

            sums["total"] += total.item() * bsz
            sums["latent_mse"] += loss_latent.item() * bsz
            sums["obs_mse"] += loss_obs.item() * bsz
            sums["obs_mae"] += obs_mae.item() * bsz
            sums["recon_mse"] += loss_recon.item() * bsz

        train_metrics = {key: value / max(total_count, 1) for key, value in sums.items()}
        val_metrics = evaluate(args, model, autoencoder, val_loader, device)

        row = {
            "epoch": epoch,
            "teacher_forcing_ratio": float(teacher_ratio),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} tf {teacher_ratio:.3f} | "
            f"train latent {train_metrics['latent_mse']:.6f} "
            f"obs {train_metrics['obs_mse']:.6f}/{train_metrics['obs_mae']:.6f} "
            f"recon {train_metrics['recon_mse']:.6f} | "
            f"val latent {val_metrics['latent_mse']:.6f} "
            f"obs {val_metrics['obs_mse']:.6f}/{val_metrics['obs_mae']:.6f} "
            f"recon {val_metrics['recon_mse']:.6f}",
            flush=True,
        )

        metric = val_metrics[args.early_stop_metric]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, args, model, autoencoder, epoch, val_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    load_checkpoint(best_path, model, autoencoder, device)

    val_metrics = evaluate(args, model, autoencoder, val_loader, device)
    test_metrics = evaluate(args, model, autoencoder, test_loader, device)

    summary = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "old_oneshot_smoke": {
            "mse": 0.383632,
            "mae": 0.396547,
        },
        "old_rollout_frozen_ae": {
            "mse": 0.3954,
            "mae": 0.4133,
        },
        "forecast_aware_ae_rollout": {
            "mse": 0.3891,
            "mae": 0.4009,
        },
    }

    with open(os.path.join(args.output_dir, "latent_rollout_tune_ae_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nLatent rollout tune-AE [test]")
    print(f"  obs MSE/MAE: {test_metrics['obs_mse']:.6f} / {test_metrics['obs_mae']:.6f}")
    print(f"  latent MSE:  {test_metrics['latent_mse']:.6f}")
    print(f"  recon MSE:   {test_metrics['recon_mse']:.6f}")
    print("  old one-shot smoke:          0.383632 / 0.396547")
    print("  old frozen-AE rollout:       0.395400 / 0.413300")
    print("  forecast-aware AE rollout:   0.389100 / 0.400900")

    extract_latents(args, autoencoder, device)


def main():
    parser = argparse.ArgumentParser(description="LatentTSF-style rollout with optional AE fine-tuning")

    parser.add_argument("--output_dir", type=str, default="./checkpoints/latent_rollout_tune_ae")
    parser.add_argument("--latent_output_dir", type=str, default="./latent_outputs/latent_rollout_tune_ae")
    parser.add_argument("--autoencoder_path", type=str, required=True)

    parser.add_argument("--model", type=str, default="DLinear")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--num_rollout_steps", type=int, default=4)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ae_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_obs", type=float, default=0.0)
    parser.add_argument("--lambda_recon", type=float, default=0.0)

    parser.add_argument("--tune_autoencoder", action="store_true", default=False)

    parser.add_argument("--teacher_forcing_start", type=float, default=0.0)
    parser.add_argument("--teacher_forcing_end", type=float, default=0.0)
    parser.add_argument("--teacher_decay_epochs", type=int, default=20)

    parser.add_argument("--early_stop_metric", type=str, default="obs_mse", choices=["latent_mse", "obs_mse", "recon_mse", "total"])

    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()