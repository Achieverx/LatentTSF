import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_utils import model_dict
from utils.latent_script_utils import (
    load_autoencoder,
    load_forecaster_from_checkpoint,
    official_loader,
    set_seed,
)


def compute_block_ranges(pred_len, num_blocks):
    num_blocks = max(1, min(int(num_blocks), int(pred_len)))
    base = pred_len // num_blocks
    ranges = []
    start = 0
    for block_idx in range(num_blocks):
        end = pred_len if block_idx == num_blocks - 1 else start + base
        ranges.append((start, end))
        start = end
    return ranges


def format_block_values(values):
    return " | ".join(f"b{idx}:{value:.4f}" for idx, value in enumerate(values))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class LatentForecaster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        backbone_args = SimpleNamespace(**vars(args))
        backbone_args.enc_in = args.d_model
        backbone_args.dec_in = args.d_model
        backbone_args.c_out = args.d_model
        self.backbone = model_dict[args.model].Model(backbone_args).float()

    def forward(self, z_x):
        z_pred = self.backbone(z_x, None, None, None)
        return z_pred[:, -self.pred_len :, :]


class LowRankLatentCalibrator(nn.Module):
    def __init__(self, pred_len, d_model, calib_type="lowrank", rank=4, num_blocks=4, alpha=0.02):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.calib_type = calib_type
        self.rank = rank
        self.num_blocks = num_blocks
        self.alpha = alpha

        if calib_type == "global":
            self.delta_param = nn.Parameter(torch.zeros(1, pred_len, d_model))
        elif calib_type == "lowrank":
            self.A = nn.Parameter(torch.randn(pred_len, rank) * 0.01)
            self.B = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        elif calib_type == "block":
            self.block_delta = nn.Parameter(torch.zeros(num_blocks, d_model))
            block_ids = torch.empty(pred_len, dtype=torch.long)
            for block_idx, (start, end) in enumerate(compute_block_ranges(pred_len, num_blocks)):
                block_ids[start:end] = block_idx
            self.register_buffer("block_ids", block_ids, persistent=False)
        else:
            raise ValueError(f"Unknown calib_type: {calib_type}")

    def raw_delta(self):
        if self.calib_type == "global":
            return self.delta_param.squeeze(0)
        if self.calib_type == "lowrank":
            return self.A @ self.B
        if self.calib_type == "block":
            return self.block_delta[self.block_ids]
        raise RuntimeError("Unsupported calib_type")

    def forward(self, z_base):
        raw_delta = self.raw_delta()
        delta = torch.tanh(raw_delta)
        z_hat = z_base + self.alpha * delta.unsqueeze(0)
        diagnostics = {
            "delta_abs_mean": delta.abs().mean(),
            "delta_abs_max": delta.abs().max(),
            "delta_norm": delta.pow(2).mean(),
            "delta_smooth": (delta[1:] - delta[:-1]).pow(2).mean() if delta.size(0) > 1 else delta.new_zeros(()),
        }
        return z_hat, delta, diagnostics


def detach_metrics(metrics):
    detached = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                detached[key] = float(value.detach().item())
            else:
                detached[key] = value.detach().cpu().tolist()
        else:
            detached[key] = value
    return detached


def compute_losses(args, y_hat, y, z_hat, z_base, delta):
    loss_forecast = F.mse_loss(y_hat, y)
    loss_anchor = F.mse_loss(z_hat, z_base.detach())
    loss_norm = delta.pow(2).mean()
    loss_smooth = (delta[1:] - delta[:-1]).pow(2).mean() if delta.size(0) > 1 else delta.new_zeros(())
    loss_total = (
        loss_forecast
        + args.lambda_anchor * loss_anchor
        + args.lambda_norm * loss_norm
        + args.lambda_smooth * loss_smooth
    )
    return {
        "loss_forecast": loss_forecast,
        "loss_anchor": loss_anchor,
        "loss_norm": loss_norm,
        "loss_smooth": loss_smooth,
        "loss_total": loss_total,
    }


def evaluate(args, autoencoder, base_model, calibrator, loader, device, block_ranges):
    calibrator.eval()
    sums = {
        "base_obs_mse": 0.0,
        "base_obs_mae": 0.0,
        "cal_obs_mse": 0.0,
        "cal_obs_mae": 0.0,
        "obs_gain": 0.0,
        "base_latent_mse": 0.0,
        "cal_latent_mse": 0.0,
        "latent_gain": 0.0,
        "anchor_dist": 0.0,
        "delta_abs_mean": 0.0,
        "delta_abs_max": 0.0,
        "delta_norm_metric": 0.0,
        "delta_smooth_metric": 0.0,
        "relative_z_change": 0.0,
        "loss_forecast": 0.0,
        "loss_anchor": 0.0,
        "loss_norm": 0.0,
        "loss_smooth": 0.0,
        "loss_total": 0.0,
    }
    count = 0
    block_base = [0.0 for _ in block_ranges]
    block_cal = [0.0 for _ in block_ranges]
    block_gain = [0.0 for _ in block_ranges]

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            z_x = autoencoder.encode(x)
            z_y = autoencoder.encode(y)
            z_base = base_model(z_x)
            y_base = autoencoder.decode(z_base)

            z_hat, delta, diag = calibrator(z_base.detach())
            y_hat = autoencoder.decode(z_hat)
            losses = compute_losses(args, y_hat, y, z_hat, z_base, delta)

            base_obs_mse = F.mse_loss(y_base, y)
            base_obs_mae = F.l1_loss(y_base, y)
            cal_obs_mse = F.mse_loss(y_hat, y)
            cal_obs_mae = F.l1_loss(y_hat, y)
            obs_gain = base_obs_mse - cal_obs_mse
            base_latent_mse = F.mse_loss(z_base, z_y)
            cal_latent_mse = F.mse_loss(z_hat, z_y)
            latent_gain = base_latent_mse - cal_latent_mse
            anchor_dist = F.mse_loss(z_hat, z_base.detach())
            relative_z_change = (z_hat - z_base).norm() / z_base.norm().clamp_min(1e-9)

            bsz = x.size(0)
            count += bsz
            for key, value in [
                ("base_obs_mse", base_obs_mse.item()),
                ("base_obs_mae", base_obs_mae.item()),
                ("cal_obs_mse", cal_obs_mse.item()),
                ("cal_obs_mae", cal_obs_mae.item()),
                ("obs_gain", obs_gain.item()),
                ("base_latent_mse", base_latent_mse.item()),
                ("cal_latent_mse", cal_latent_mse.item()),
                ("latent_gain", latent_gain.item()),
                ("anchor_dist", anchor_dist.item()),
                ("delta_abs_mean", diag["delta_abs_mean"].item()),
                ("delta_abs_max", diag["delta_abs_max"].item()),
                ("delta_norm_metric", diag["delta_norm"].item()),
                ("delta_smooth_metric", diag["delta_smooth"].item()),
                ("relative_z_change", relative_z_change.item()),
                ("loss_forecast", losses["loss_forecast"].item()),
                ("loss_anchor", losses["loss_anchor"].item()),
                ("loss_norm", losses["loss_norm"].item()),
                ("loss_smooth", losses["loss_smooth"].item()),
                ("loss_total", losses["loss_total"].item()),
            ]:
                sums[key] += value * bsz

            base_sq = (y_base - y).pow(2)
            cal_sq = (y_hat - y).pow(2)
            for idx, (start, end) in enumerate(block_ranges):
                base_block_mse = base_sq[:, start:end, :].mean().item()
                cal_block_mse = cal_sq[:, start:end, :].mean().item()
                block_base[idx] += base_block_mse * bsz
                block_cal[idx] += cal_block_mse * bsz
                block_gain[idx] += (base_block_mse - cal_block_mse) * bsz

    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["block_base_mse"] = [value / max(count, 1) for value in block_base]
    result["block_cal_mse"] = [value / max(count, 1) for value in block_cal]
    result["block_gain"] = [value / max(count, 1) for value in block_gain]
    return result


def epoch_row(epoch, split, metrics, num_blocks):
    row = {
        "epoch": epoch,
        "split": split,
        "base_obs_mse": metrics["base_obs_mse"],
        "base_obs_mae": metrics["base_obs_mae"],
        "cal_obs_mse": metrics["cal_obs_mse"],
        "cal_obs_mae": metrics["cal_obs_mae"],
        "obs_gain": metrics["obs_gain"],
        "base_latent_mse": metrics["base_latent_mse"],
        "cal_latent_mse": metrics["cal_latent_mse"],
        "latent_gain": metrics["latent_gain"],
        "anchor_dist": metrics["anchor_dist"],
        "delta_abs_mean": metrics["delta_abs_mean"],
        "delta_abs_max": metrics["delta_abs_max"],
        "delta_norm_metric": metrics["delta_norm_metric"],
        "delta_smooth_metric": metrics["delta_smooth_metric"],
        "relative_z_change": metrics["relative_z_change"],
        "loss_forecast": metrics["loss_forecast"],
        "loss_anchor": metrics["loss_anchor"],
        "loss_norm": metrics["loss_norm"],
        "loss_smooth": metrics["loss_smooth"],
        "loss_total": metrics["loss_total"],
    }
    for idx in range(num_blocks):
        row[f"block_base_mse_b{idx}"] = metrics["block_base_mse"][idx]
        row[f"block_cal_mse_b{idx}"] = metrics["block_cal_mse"][idx]
        row[f"block_gain_b{idx}"] = metrics["block_gain"][idx]
    return row


def save_checkpoint(path, args, calibrator, epoch, val_metrics, test_metrics):
    torch.save(
        {
            "args": vars(args),
            "epoch": epoch,
            "calibrator_state_dict": calibrator.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


def print_epoch_line(epoch, train_metrics, val_metrics, test_metrics):
    print(
        f"epoch {epoch:03d} | "
        f"train {train_metrics['base_obs_mse']:.6f}->{train_metrics['cal_obs_mse']:.6f} (gain {train_metrics['obs_gain']:.6f}) | "
        f"val {val_metrics['base_obs_mse']:.6f}->{val_metrics['cal_obs_mse']:.6f} (gain {val_metrics['obs_gain']:.6f}) | "
        f"test {test_metrics['base_obs_mse']:.6f}->{test_metrics['cal_obs_mse']:.6f} (gain {test_metrics['obs_gain']:.6f})"
    )
    print(
        f"           latent val/test {val_metrics['base_latent_mse']:.6f}->{val_metrics['cal_latent_mse']:.6f} | "
        f"gain {val_metrics['latent_gain']:.6f}/{test_metrics['latent_gain']:.6f}"
    )
    print(
        f"           anchor {val_metrics['anchor_dist']:.6f} | rel_z {val_metrics['relative_z_change']:.6f} | "
        f"delta mean/max {val_metrics['delta_abs_mean']:.6f}/{val_metrics['delta_abs_max']:.6f}"
    )
    print(
        f"           delta norm/smooth {val_metrics['delta_norm_metric']:.6f}/{val_metrics['delta_smooth_metric']:.6f}"
    )
    print(
        f"           block gain val {format_block_values(val_metrics['block_gain'])} | "
        f"test {format_block_values(test_metrics['block_gain'])}"
    )


def train(args, device):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device, freeze=True)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    base_model, _ = load_forecaster_from_checkpoint(
        args,
        device,
        args.base_checkpoint,
        LatentForecaster,
        freeze=True,
        return_checkpoint=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    calibrator = LowRankLatentCalibrator(
        pred_len=args.pred_len,
        d_model=args.d_model,
        calib_type=args.calib_type,
        rank=args.rank,
        num_blocks=args.num_blocks,
        alpha=args.alpha,
    ).to(device)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)
    block_ranges = compute_block_ranges(args.pred_len, args.num_blocks)

    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Low-Rank Latent Calibration | model={args.model} data={args.data} pred_len={args.pred_len} "
        f"type={args.calib_type} rank={args.rank} alpha={args.alpha}"
    )
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(
        f"Trainable params: {count_trainable_params(calibrator)} | "
        f"Frozen params: {count_params(autoencoder) + count_params(base_model)}"
    )
    print(
        f"Hyperparams | lambda_anchor={args.lambda_anchor} lambda_norm={args.lambda_norm} "
        f"lambda_smooth={args.lambda_smooth} lr={args.lr}"
    )

    history_rows = []
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")

    for epoch in range(1, args.epochs + 1):
        calibrator.train()
        for batch_x, batch_y, _, _ in train_loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            with torch.no_grad():
                z_x = autoencoder.encode(x)
                z_base = base_model(z_x)

            z_hat, delta, _ = calibrator(z_base.detach())
            y_hat = autoencoder.decode(z_hat)
            losses = compute_losses(args, y_hat, y, z_hat, z_base, delta)

            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(calibrator.parameters(), args.grad_clip)
            optimizer.step()

        train_metrics = evaluate(args, autoencoder, base_model, calibrator, eval_train_loader, device, block_ranges)
        val_metrics = evaluate(args, autoencoder, base_model, calibrator, val_loader, device, block_ranges)
        test_metrics = evaluate(args, autoencoder, base_model, calibrator, test_loader, device, block_ranges)
        print_epoch_line(epoch, train_metrics, val_metrics, test_metrics)

        history_rows.append(epoch_row(epoch, "train", train_metrics, len(block_ranges)))
        history_rows.append(epoch_row(epoch, "val", val_metrics, len(block_ranges)))
        history_rows.append(epoch_row(epoch, "test", test_metrics, len(block_ranges)))
        pd.DataFrame(history_rows).to_csv(csv_path, index=False)

        current_val = val_metrics["cal_obs_mse"]
        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_checkpoint_path, args, calibrator, epoch, val_metrics, test_metrics)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

    checkpoint_data = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    calibrator.load_state_dict(checkpoint_data["calibrator_state_dict"])
    calibrator.eval()

    final_train = evaluate(args, autoencoder, base_model, calibrator, eval_train_loader, device, block_ranges)
    final_val = evaluate(args, autoencoder, base_model, calibrator, val_loader, device, block_ranges)
    final_test = evaluate(args, autoencoder, base_model, calibrator, test_loader, device, block_ranges)

    summary = {
        "best_epoch": best_epoch,
        "test_base_obs_mse": final_test["base_obs_mse"],
        "test_cal_obs_mse": final_test["cal_obs_mse"],
        "test_obs_gain": final_test["obs_gain"],
        "test_base_latent_mse": final_test["base_latent_mse"],
        "test_cal_latent_mse": final_test["cal_latent_mse"],
        "test_latent_gain": final_test["latent_gain"],
        "test_anchor_dist": final_test["anchor_dist"],
        "test_relative_z_change": final_test["relative_z_change"],
        "calib_type": args.calib_type,
        "rank": args.rank,
        "alpha": args.alpha,
        "lambda_anchor": args.lambda_anchor,
        "lambda_norm": args.lambda_norm,
        "lambda_smooth": args.lambda_smooth,
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    diagnostics = {
        "train": detach_metrics(final_train),
        "val": detach_metrics(final_val),
        "test": detach_metrics(final_test),
        "block_ranges": block_ranges,
    }
    with open(os.path.join(args.output_dir, "test_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(
        f"Low-Rank Latent Calibration [test]\n"
        f"  obs MSE / MAE: {final_test['base_obs_mse']:.6f} / {final_test['base_obs_mae']:.6f} -> "
        f"{final_test['cal_obs_mse']:.6f} / {final_test['cal_obs_mae']:.6f}\n"
        f"  obs gain: {final_test['obs_gain']:.6f}\n"
        f"  latent gain: {final_test['latent_gain']:.6f}\n"
        f"  anchor {final_test['anchor_dist']:.6f} | rel_z {final_test['relative_z_change']:.6f}\n"
        f"  block gain: {format_block_values(final_test['block_gain'])}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Low-Rank Decoder-Compatible Latent Calibration")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--dec_in", type=int, required=True)
    parser.add_argument("--c_out", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--calib_type", type=str, default="lowrank", choices=["global", "lowrank", "block"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--lambda_anchor", type=float, default=0.5)
    parser.add_argument("--lambda_norm", type=float, default=0.005)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
