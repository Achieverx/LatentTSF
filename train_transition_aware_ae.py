import argparse
import json
import os

import pandas as pd  # preload before torch to avoid pyarrow issue on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder


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


class StrongRolloutForecaster(torch.nn.Module):
    """
    Strong latent rollout forecaster.

    It keeps the full latent context sequence at every rollout step.
    Each step predicts one future latent chunk, appends it to context,
    and keeps the latest seq_len latent tokens.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        num_steps: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        shared_block: bool = True,
    ):
        super().__init__()

        assert pred_len % num_steps == 0, (
            "For this first version, pred_len must be divisible by num_steps."
        )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.num_steps = num_steps
        self.chunk_len = pred_len // num_steps
        self.shared_block = shared_block

        def make_block():
            return torch.nn.Sequential(
                torch.nn.Linear(seq_len * d_model, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, self.chunk_len * d_model),
            )

        if shared_block:
            self.block = make_block()
        else:
            self.blocks = torch.nn.ModuleList([make_block() for _ in range(num_steps)])

        self.step_emb = torch.nn.Parameter(torch.randn(num_steps, seq_len * d_model) * 0.01)

    def forward(self, z_x):
        context = z_x
        chunks = []

        for step in range(self.num_steps):
            bsz = context.size(0)

            flat_context = context.reshape(bsz, self.seq_len * self.d_model)
            flat_context = flat_context + self.step_emb[step].unsqueeze(0)

            if self.shared_block:
                chunk = self.block(flat_context)
            else:
                chunk = self.blocks[step](flat_context)

            chunk = chunk.view(bsz, self.chunk_len, self.d_model)
            chunks.append(chunk)

            context = torch.cat([context, chunk], dim=1)
            context = context[:, -self.seq_len:, :]

        return torch.cat(chunks, dim=1)


def encode_batch(autoencoder, batch_x, batch_y, pred_len, device, freeze_autoencoder=False):
    x = batch_x.float().to(device)
    y = batch_y[:, -pred_len:, :].float().to(device)

    if freeze_autoencoder:
        with torch.no_grad():
            z_x = autoencoder.encode(x)
            z_y = autoencoder.encode(y)
    else:
        z_x = autoencoder.encode(x)
        z_y = autoencoder.encode(y)

    return x, y, z_x, z_y


def compute_losses(autoencoder, forecaster, batch_x, batch_y, args, device):
    _, y, z_x, z_y = encode_batch(
        autoencoder,
        batch_x,
        batch_y,
        args.pred_len,
        device,
        freeze_autoencoder=args.freeze_autoencoder,
    )

    if args.freeze_autoencoder:
        with torch.no_grad():
            y_recon = autoencoder.decode(z_y)
    else:
        y_recon = autoencoder.decode(z_y)

    loss_recon = F.mse_loss(y_recon, y)

    z_pred = forecaster(z_x)

    loss_latent = F.mse_loss(z_pred, z_y.detach())

    # If AE is frozen, gradients still flow through decoder operation into z_pred.
    # AE params do not update because requires_grad=False.
    y_pred = autoencoder.decode(z_pred)

    loss_obs = F.mse_loss(y_pred, y)
    mae = (y_pred - y).abs().mean()

    total = (
        args.lambda_recon * loss_recon
        + args.lambda_latent * loss_latent
        + args.lambda_obs * loss_obs
    )

    return total, loss_recon, loss_latent, loss_obs, mae


def evaluate(args, autoencoder, forecaster, loader, device):
    autoencoder.eval()
    forecaster.eval()

    totals = {
        "total": 0.0,
        "recon": 0.0,
        "latent": 0.0,
        "obs": 0.0,
        "mae": 0.0,
    }
    count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            total, recon, latent, obs, mae = compute_losses(
                autoencoder, forecaster, batch_x, batch_y, args, device
            )

            bsz = batch_x.size(0)
            count += bsz

            totals["total"] += total.item() * bsz
            totals["recon"] += recon.item() * bsz
            totals["latent"] += latent.item() * bsz
            totals["obs"] += obs.item() * bsz
            totals["mae"] += mae.item() * bsz

    return {k: v / max(count, 1) for k, v in totals.items()}


def save_checkpoint(path, args, autoencoder, forecaster, epoch, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "autoencoder_state_dict": autoencoder.state_dict(),
            "forecaster_state_dict": forecaster.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_best(path, autoencoder, forecaster, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    forecaster.load_state_dict(checkpoint["forecaster_state_dict"])
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


def build_optimizer(args, autoencoder, forecaster):
    if args.freeze_autoencoder:
        for p in autoencoder.parameters():
            p.requires_grad = False

        params = list(forecaster.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return optimizer, params

    # Optional: separate AE lr and forecaster lr.
    params = [
        {"params": forecaster.parameters(), "lr": args.lr},
        {"params": autoencoder.parameters(), "lr": args.ae_lr},
    ]

    grad_params = list(forecaster.parameters()) + list(autoencoder.parameters())

    optimizer = torch.optim.AdamW(
        params,
        weight_decay=args.weight_decay,
    )

    return optimizer, grad_params


def train(args, device):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.latent_output_dir, exist_ok=True)

    with open(os.path.join(args.checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device).to(device)

    forecaster = StrongRolloutForecaster(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        num_steps=args.num_steps,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        shared_block=args.shared_block,
    ).to(device)

    optimizer, grad_params = build_optimizer(args, autoencoder, forecaster)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    best_path = os.path.join(args.checkpoint_dir, "checkpoint.pth")
    best_val = float("inf")
    best_epoch = 0
    history = []

    print(
        f"Training mode | freeze_autoencoder={args.freeze_autoencoder} "
        f"lr={args.lr} ae_lr={args.ae_lr} "
        f"lambda_recon={args.lambda_recon} "
        f"lambda_latent={args.lambda_latent} "
        f"lambda_obs={args.lambda_obs}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        if args.freeze_autoencoder:
            autoencoder.eval()
        else:
            autoencoder.train()

        forecaster.train()

        sums = {
            "total": 0.0,
            "recon": 0.0,
            "latent": 0.0,
            "obs": 0.0,
            "mae": 0.0,
        }
        count = 0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            total, recon, latent, obs, mae = compute_losses(
                autoencoder, forecaster, batch_x, batch_y, args, device
            )

            optimizer.zero_grad()
            total.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)

            optimizer.step()

            bsz = batch_x.size(0)
            count += bsz

            sums["total"] += total.item() * bsz
            sums["recon"] += recon.item() * bsz
            sums["latent"] += latent.item() * bsz
            sums["obs"] += obs.item() * bsz
            sums["mae"] += mae.item() * bsz

        train_metrics = {k: v / max(count, 1) for k, v in sums.items()}
        val_metrics = evaluate(args, autoencoder, forecaster, val_loader, device)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch {epoch:03d} | "
            f"train total {train_metrics['total']:.6f} "
            f"recon {train_metrics['recon']:.6f} "
            f"latent {train_metrics['latent']:.6f} "
            f"obs {train_metrics['obs']:.6f}/{train_metrics['mae']:.6f} | "
            f"val total {val_metrics['total']:.6f} "
            f"recon {val_metrics['recon']:.6f} "
            f"latent {val_metrics['latent']:.6f} "
            f"obs {val_metrics['obs']:.6f}/{val_metrics['mae']:.6f}",
            flush=True,
        )

        if val_metrics[args.early_stop_metric] < best_val:
            best_val = val_metrics[args.early_stop_metric]
            best_epoch = epoch
            save_checkpoint(best_path, args, autoencoder, forecaster, epoch, val_metrics)

    load_best(best_path, autoencoder, forecaster, device)

    train_metrics = evaluate(args, autoencoder, forecaster, train_loader, device)
    val_metrics = evaluate(args, autoencoder, forecaster, val_loader, device)
    test_metrics = evaluate(args, autoencoder, forecaster, test_loader, device)

    summary = {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "old_oneshot_smoke": {
            "mse": 0.3836321234703064,
            "mae": 0.39654698967933655,
        },
        "old_rollout": {
            "mse": 0.3954,
            "mae": 0.4133,
        },
        "forecast_aware_ae_rollout": {
            "mse": 0.3891,
            "mae": 0.4009,
        },
    }

    with open(os.path.join(args.checkpoint_dir, "transition_aware_ae_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTransition-aware AE rollout smoke [test]")
    print(f"  MSE/MAE: {test_metrics['obs']:.6f} / {test_metrics['mae']:.6f}")
    print(f"  recon loss: {test_metrics['recon']:.6f}")
    print(f"  latent rollout loss: {test_metrics['latent']:.6f}")
    print("  old one-shot smoke: 0.383632 / 0.396547")
    print("  old pure rollout:    0.395400 / 0.413300")
    print("  forecast-aware AE rollout: 0.389100 / 0.400900")

    extract_latents(args, autoencoder, device)


def main():
    parser = argparse.ArgumentParser(description="Transition-aware AE fine-tuning with strong latent rollout")

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/transition_aware_ae_ETTh1_sl96_pl96")
    parser.add_argument("--latent_output_dir", type=str, default="./latent_outputs/transition_aware_ae_ETTh1_sl96_pl96")
    parser.add_argument("--autoencoder_path", type=str, required=True)

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
    parser.add_argument("--step", type=int, default=1)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)

    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shared_block", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ae_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_recon", type=float, default=0.0)
    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_obs", type=float, default=0.1)

    parser.add_argument("--freeze_autoencoder", action="store_true", default=False)

    parser.add_argument("--early_stop_metric", type=str, default="obs", choices=["total", "recon", "latent", "obs"])
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()