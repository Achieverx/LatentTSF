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
from train_latent_proto_regularized import LatentForecaster


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
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    return autoencoder


def load_latenttsf(args, device):
    checkpoint = torch.load(args.latenttsf_checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    base_args = SimpleNamespace(**vars(args))
    for key, value in ckpt_args.items():
        setattr(base_args, key, value)
    model = LatentForecaster(base_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class ResidualAdapter(torch.nn.Module):
    def __init__(self, channels, hidden_dim, dropout, alpha_init, alpha_max):
        super().__init__()
        self.alpha_max = alpha_max
        init_ratio = min(max(alpha_init / alpha_max, 1e-6), 1 - 1e-6)
        self.alpha_logit = torch.nn.Parameter(torch.tensor(np.log(init_ratio / (1 - init_ratio)), dtype=torch.float32))
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(channels * 4),
            torch.nn.Linear(channels * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, channels),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def forward(self, x, y_base):
        last = x[:, -1:, :].expand(-1, y_base.size(1), -1)
        mean = x.mean(dim=1, keepdim=True).expand_as(last)
        std = x.std(dim=1, keepdim=True, unbiased=False).expand_as(last)
        features = torch.cat([y_base, last, mean, std], dim=-1)
        delta = self.net(features)
        return y_base + self.alpha() * delta, delta


def baseline_forecast(autoencoder, latenttsf, x):
    with torch.no_grad():
        z_x = autoencoder.encode(x)
        z_base = latenttsf(z_x)
        y_base = autoencoder.decode(z_base)
    return y_base


def mse_mae(pred, target):
    return F.mse_loss(pred, target), (pred - target).abs().mean()


def evaluate(args, adapter, autoencoder, latenttsf, loader, device):
    adapter.eval()
    sums = {
        "base_mse": 0.0,
        "base_mae": 0.0,
        "pred_mse": 0.0,
        "pred_mae": 0.0,
        "gain": 0.0,
        "harmful_ratio": 0.0,
        "delta_abs": 0.0,
        "applied_delta_abs": 0.0,
    }
    total = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)
            y_base = baseline_forecast(autoencoder, latenttsf, x)
            y_hat, delta = adapter(x, y_base)
            base_sample = ((y_base - y) ** 2).mean(dim=(1, 2))
            pred_sample = ((y_hat - y) ** 2).mean(dim=(1, 2))
            bsz = x.size(0)
            total += bsz
            sums["base_mse"] += base_sample.sum().item()
            sums["base_mae"] += (y_base - y).abs().mean(dim=(1, 2)).sum().item()
            sums["pred_mse"] += pred_sample.sum().item()
            sums["pred_mae"] += (y_hat - y).abs().mean(dim=(1, 2)).sum().item()
            sums["gain"] += (base_sample - pred_sample).sum().item()
            sums["harmful_ratio"] += (pred_sample > base_sample).float().sum().item()
            sums["delta_abs"] += delta.abs().mean(dim=(1, 2)).sum().item()
            sums["applied_delta_abs"] += (adapter.alpha() * delta).abs().mean(dim=(1, 2)).sum().item()
    result = {key: value / max(total, 1) for key, value in sums.items()}
    result["alpha"] = adapter.alpha().item()
    return result


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    latenttsf = load_latenttsf(args, device)
    adapter = ResidualAdapter(args.enc_in, args.adapter_hidden_dim, args.dropout, args.alpha_init, args.alpha_max).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    print(
        f"LatentTSF residual adapter | frozen baseline | lr={args.lr} "
        f"alpha_init={args.alpha_init} alpha_max={args.alpha_max}",
        flush=True,
    )
    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)
            y_base = baseline_forecast(autoencoder, latenttsf, x)
            y_hat, delta = adapter(x, y_base)
            loss = F.mse_loss(y_hat, y)
            if args.lambda_delta_l2 > 0:
                loss = loss + args.lambda_delta_l2 * delta.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)

        train_metrics = evaluate(args, adapter, autoencoder, latenttsf, train_loader, device)
        val_metrics = evaluate(args, adapter, autoencoder, latenttsf, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | train pred/base {train_metrics['pred_mse']:.6f}/{train_metrics['base_mse']:.6f} "
            f"gain {train_metrics['gain']:.6f} | val pred/base {val_metrics['pred_mse']:.6f}/{val_metrics['base_mse']:.6f} "
            f"gain {val_metrics['gain']:.6f} harm {val_metrics['harmful_ratio']:.3f} alpha {val_metrics['alpha']:.4f}",
            flush=True,
        )

        metric = val_metrics["pred_mse"]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
            torch.save(
                {
                    "adapter_state_dict": best_state,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "best_residual_adapter.pt"),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    train_metrics = evaluate(args, adapter, autoencoder, latenttsf, train_loader, device)
    val_metrics = evaluate(args, adapter, autoencoder, latenttsf, val_loader, device)
    test_metrics = evaluate(args, adapter, autoencoder, latenttsf, test_loader, device)
    summary = {
        "best_epoch": best_epoch,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(args.output_dir, "residual_adapter_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nLatentTSF residual adapter [test]")
    print(f"  baseline MSE/MAE: {test_metrics['base_mse']:.6f} / {test_metrics['base_mae']:.6f}")
    print(f"  adapter  MSE/MAE: {test_metrics['pred_mse']:.6f} / {test_metrics['pred_mae']:.6f}")
    print(f"  gain vs baseline: {test_metrics['gain']:.6f}")
    print(f"  harmful ratio:    {test_metrics['harmful_ratio']:.4f}")
    print(f"  alpha:            {test_metrics['alpha']:.6f}")
    print(f"  |delta|/|applied|:{test_metrics['delta_abs']:.6f} / {test_metrics['applied_delta_abs']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Frozen LatentTSF plus zero-initialized residual adapter")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/latenttsf_residual_adapter")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--latenttsf_checkpoint", type=str, required=True)
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
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--adapter_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha_init", type=float, default=0.01)
    parser.add_argument("--alpha_max", type=float, default=0.2)
    parser.add_argument("--lambda_delta_l2", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
