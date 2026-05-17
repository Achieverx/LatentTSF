import argparse
import json
import os

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_latent_proto_regularized import LatentForecaster
from utils.latent_script_utils import (
    load_autoencoder,
    load_forecaster_from_checkpoint,
    merge_checkpoint_args,
    official_loader,
    set_seed,
)
from utils.metrics import metric


CHECKPOINT_ARG_KEYS = [
    "model",
    "task_name",
    "data",
    "root_path",
    "data_path",
    "features",
    "target",
    "freq",
    "embed",
    "seasonal_patterns",
    "seq_len",
    "label_len",
    "pred_len",
    "step",
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "d_ff",
    "n_heads",
    "e_layers",
    "d_layers",
    "factor",
    "activation",
    "moving_avg",
    "individual",
    "dropout",
    "top_k",
    "num_kernels",
    "patch_len",
    "channel_independence",
    "decomp_method",
    "use_norm",
    "down_sampling_layers",
    "down_sampling_window",
    "ae_type",
    "batch_size",
    "num_workers",
    "inverse",
    "augmentation_ratio",
]

METRICS_FILENAMES = [
    "latent_proto_regularized_metrics.json",
    "metrics.json",
]

ABLATION_CHOICES = [
    "zero_adapter",
    "full",
    "random_mismatch",
    "no_corr_loss",
    "old_output_adapter",
]


def sample_corr(series, eps):
    centered = series - series.mean(dim=1, keepdim=True)
    denom = torch.sqrt(centered.pow(2).mean(dim=1, keepdim=True) + eps)
    normalized = centered / denom
    corr = torch.einsum("btc,btd->bcd", normalized, normalized) / max(series.size(1), 1)
    eye = torch.eye(series.size(-1), device=series.device, dtype=series.dtype).unsqueeze(0)
    return corr * (1.0 - eye)


class CorrelationMismatchCalibrationAdapter(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        dropout,
        alpha,
        mode="full",
        corr_eps=1e-6,
        pool_gate_hidden_dim=None,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.alpha = float(alpha)
        self.mode = mode
        self.corr_eps = float(corr_eps)

        pool_hidden = hidden_dim if pool_gate_hidden_dim is None else pool_gate_hidden_dim

        self.channel_mixer = nn.Parameter(torch.zeros(channels, channels))

        self.mismatch_gate = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
        )
        self.pool_gate = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Linear(channels * 3, pool_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_hidden, channels),
        )

        nn.init.zeros_(self.channel_mixer)

    def _mismatch_features(self, batch_x, y_base):
        corr_x = sample_corr(batch_x, self.corr_eps)
        corr_base = sample_corr(y_base.detach(), self.corr_eps)
        mismatch = corr_x - corr_base
        mismatch_abs = mismatch.abs()
        mismatch_mean = mismatch_abs.mean(dim=-1)
        mismatch_max = mismatch_abs.max(dim=-1).values
        mismatch_features = torch.cat([mismatch_mean, mismatch_max], dim=-1)
        return corr_x, corr_base, mismatch, mismatch_features

    def _pool_features(self, batch_x):
        return torch.cat(
            [
                batch_x.mean(dim=1),
                batch_x.std(dim=1, unbiased=False),
                batch_x[:, -1, :],
            ],
            dim=-1,
        )

    def forward(self, batch_x, y_base):
        corr_x, corr_base, mismatch, mismatch_features = self._mismatch_features(batch_x, y_base)
        used_mismatch = mismatch_features

        if self.mode == "random_mismatch" and mismatch_features.size(0) > 1:
            if self.training:
                perm = torch.randperm(mismatch_features.size(0), device=mismatch_features.device)
            else:
                perm = torch.roll(torch.arange(mismatch_features.size(0), device=mismatch_features.device), shifts=1)
            used_mismatch = mismatch_features[perm]

        if self.mode == "zero_adapter":
            gate = torch.zeros(
                batch_x.size(0),
                self.channels,
                device=batch_x.device,
                dtype=batch_x.dtype,
            )
            correction = torch.zeros_like(y_base)
            y_hat = y_base
        elif self.mode == "old_output_adapter":
            gate = torch.sigmoid(self.pool_gate(self._pool_features(batch_x)))
            delta = torch.matmul(y_base, self.channel_mixer)
            correction = self.alpha * gate[:, None, :] * delta
            y_hat = y_base + correction
        else:
            gate = torch.sigmoid(self.mismatch_gate(used_mismatch))
            delta = torch.matmul(y_base, self.channel_mixer)
            correction = self.alpha * gate[:, None, :] * delta
            y_hat = y_base + correction

        corr_hat = sample_corr(y_hat, self.corr_eps)
        correction_abs_mean = correction.abs().mean()
        relative_correction = correction_abs_mean / y_base.abs().mean().clamp_min(self.corr_eps)
        stats = {
            "gate_mean": gate.mean(),
            "gate_std": gate.std(unbiased=False),
            "correction_abs_mean": correction_abs_mean,
            "relative_correction": relative_correction,
            "mismatch_abs_mean": mismatch.abs().mean(),
            "corr_x_abs_mean": corr_x.abs().mean(),
            "corr_base_abs_mean": corr_base.abs().mean(),
        }
        return y_hat, correction, gate, corr_x, corr_hat, stats


def load_latenttsf(args, device):
    return load_forecaster_from_checkpoint(
        args,
        device,
        args.latenttsf_checkpoint,
        LatentForecaster,
        return_checkpoint=True,
    )


def freeze_and_assert(module, name):
    module.eval()
    module.requires_grad_(False)
    if any(param.requires_grad for param in module.parameters()):
        raise RuntimeError(f"{name} is not fully frozen.")


def load_expected_metrics(args, checkpoint_path):
    if args.expected_test_mse is not None:
        return {
            "mse": float(args.expected_test_mse),
            "mae": None if args.expected_test_mae is None else float(args.expected_test_mae),
            "source": "cli",
        }

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    for filename in METRICS_FILENAMES:
        metrics_path = os.path.join(checkpoint_dir, filename)
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        test_metrics = payload.get("test", payload)
        if "pred_mse" in test_metrics:
            return {
                "mse": float(test_metrics["pred_mse"]),
                "mae": None if "pred_mae" not in test_metrics else float(test_metrics["pred_mae"]),
                "source": metrics_path,
            }
        if "test_mse" in test_metrics:
            return {
                "mse": float(test_metrics["test_mse"]),
                "mae": None if "test_mae" not in test_metrics else float(test_metrics["test_mae"]),
                "source": metrics_path,
            }
    return None


def prepare_args(args, checkpoint):
    checkpoint_args = checkpoint.get("args", {})
    overwritten = merge_checkpoint_args(args, checkpoint_args, CHECKPOINT_ARG_KEYS)
    return checkpoint_args, overwritten


def inverse_transform_array(dataset, pred, target_shape):
    pred_np = pred
    if pred_np.shape[-1] != target_shape[-1]:
        pred_np = np.tile(pred_np, [1, 1, int(target_shape[-1] / pred_np.shape[-1])])
    flat = pred_np.reshape(target_shape[0] * target_shape[1], -1)
    return dataset.inverse_transform(flat).reshape(target_shape)


def forward_mismatch_batch(args, autoencoder, latenttsf, adapter, batch_x, batch_y):
    x = batch_x.float()
    y = batch_y.float()
    y_true_full = y[:, -args.pred_len :, :]

    with torch.no_grad():
        z_x = autoencoder.encode(x)
        z_pred_base = latenttsf(z_x)
        y_base_full = autoencoder.decode(z_pred_base)

    y_base_full = y_base_full[:, -args.pred_len :, :]
    y_hat_full, correction_full, gate, corr_x, corr_hat, adapter_stats = adapter(x, y_base_full)

    f_dim = -1 if args.features == "MS" else 0
    y_base = y_base_full[:, :, f_dim:]
    y_hat = y_hat_full[:, :, f_dim:]
    y_true = y_true_full[:, :, f_dim:]
    return {
        "x": x,
        "y_true": y_true,
        "y_base": y_base,
        "y_hat": y_hat,
        "y_base_full": y_base_full,
        "y_hat_full": y_hat_full,
        "correction_full": correction_full,
        "gate": gate,
        "corr_x": corr_x,
        "corr_hat": corr_hat,
        "adapter_stats": adapter_stats,
    }


def evaluate_loader(args, adapter, autoencoder, latenttsf, dataset, loader, device):
    adapter.eval()
    preds_base = []
    preds_adapter = []
    trues = []
    sums = {
        "base_loss": 0.0,
        "pred_loss": 0.0,
        "reg_loss": 0.0,
        "corr_loss": 0.0,
        "gate_mean": 0.0,
        "gate_std": 0.0,
        "correction_abs_mean": 0.0,
        "relative_correction": 0.0,
        "mismatch_abs_mean": 0.0,
        "corr_x_abs_mean": 0.0,
        "corr_base_abs_mean": 0.0,
    }
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().to(device)
            y = batch_y.float().to(device)
            batch = forward_mismatch_batch(args, autoencoder, latenttsf, adapter, x, y)

            base_loss = F.mse_loss(batch["y_base"], batch["y_true"])
            pred_loss = F.mse_loss(batch["y_hat"], batch["y_true"])
            reg_loss = (batch["y_hat_full"] - batch["y_base_full"]).pow(2).mean()
            corr_loss = F.mse_loss(batch["corr_hat"], batch["corr_x"])

            bsz = x.size(0)
            total_count += bsz
            sums["base_loss"] += base_loss.item() * bsz
            sums["pred_loss"] += pred_loss.item() * bsz
            sums["reg_loss"] += reg_loss.item() * bsz
            sums["corr_loss"] += corr_loss.item() * bsz
            for key in [
                "gate_mean",
                "gate_std",
                "correction_abs_mean",
                "relative_correction",
                "mismatch_abs_mean",
                "corr_x_abs_mean",
                "corr_base_abs_mean",
            ]:
                sums[key] += batch["adapter_stats"][key].item() * bsz

            y_base_np = batch["y_base"].detach().cpu().numpy()
            y_hat_np = batch["y_hat"].detach().cpu().numpy()
            y_true_np = batch["y_true"].detach().cpu().numpy()
            if getattr(dataset, "scale", False) and args.inverse:
                shape = y_true_np.shape
                y_base_np = inverse_transform_array(dataset, y_base_np, shape)
                y_hat_np = inverse_transform_array(dataset, y_hat_np, shape)
                y_true_np = dataset.inverse_transform(y_true_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

            preds_base.append(y_base_np)
            preds_adapter.append(y_hat_np)
            trues.append(y_true_np)

    preds_base = np.concatenate(preds_base, axis=0)
    preds_adapter = np.concatenate(preds_adapter, axis=0)
    trues = np.concatenate(trues, axis=0)

    base_mae, base_mse, _, _, _ = metric(preds_base, trues)
    pred_mae, pred_mse, _, _, _ = metric(preds_adapter, trues)
    result = {
        "base_mse": base_mse,
        "base_mae": base_mae,
        "adapter_mse": pred_mse,
        "adapter_mae": pred_mae,
        "gain_mse": base_mse - pred_mse,
        "gain_mae": base_mae - pred_mae,
    }
    for key, value in sums.items():
        result[key] = value / max(total_count, 1)
    return result


def print_split_metrics(split_name, split_metrics):
    print(
        f"{split_name} | "
        f"baseline MSE / MAE {split_metrics['base_mse']:.6f} / {split_metrics['base_mae']:.6f} | "
        f"adapter MSE / MAE {split_metrics['adapter_mse']:.6f} / {split_metrics['adapter_mae']:.6f} | "
        f"gain {split_metrics['gain_mse']:.6f} / {split_metrics['gain_mae']:.6f}",
        flush=True,
    )
    print(
        f"       gate mean/std {split_metrics['gate_mean']:.6f}/{split_metrics['gate_std']:.6f} | "
        f"corr_loss {split_metrics['corr_loss']:.6f} | "
        f"correction_abs_mean {split_metrics['correction_abs_mean']:.6f} | "
        f"relative_correction {split_metrics['relative_correction']:.6f}",
        flush=True,
    )


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    latenttsf, checkpoint = load_latenttsf(args, device)
    checkpoint_args, overwritten = prepare_args(args, checkpoint)
    autoencoder = load_autoencoder(args, device)

    freeze_and_assert(autoencoder, "AE")
    freeze_and_assert(latenttsf, "LatentTSF")

    adapter = CorrelationMismatchCalibrationAdapter(
        channels=args.enc_in,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        mode=args.adapter_mode,
        corr_eps=args.corr_eps,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)
    train_data = eval_train_loader.dataset
    val_data = val_loader.dataset
    test_data = test_loader.dataset

    expected_metrics = load_expected_metrics(args, args.latenttsf_checkpoint)

    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "checkpoint_args": checkpoint_args,
                "checkpoint_overrides": overwritten,
                "expected_metrics": expected_metrics,
            },
            f,
            indent=2,
        )

    best_state = None
    best_val_mse = float("inf")
    best_val_base_mse = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"Correlation-Mismatch Calibration Adapter | mode={args.adapter_mode} "
        f"lr={args.lr} alpha={args.alpha} lambda_reg={args.lambda_reg} "
        f"lambda_corr={args.lambda_corr} hidden={args.adapter_hidden_dim}",
        flush=True,
    )
    if args.adapter_mode == "zero_adapter":
        print("Zero-adapter mode: skip parameter updates and use it as a baseline-consistency ablation.", flush=True)
    if overwritten:
        print("Checkpoint-aligned args:", flush=True)
        for key, values in overwritten.items():
            print(f"  {key}: {values['old']} -> {values['new']}", flush=True)

    corr_weight = 0.0 if args.adapter_mode == "no_corr_loss" else args.lambda_corr

    for epoch in range(1, args.epochs + 1):
        if args.adapter_mode != "zero_adapter":
            adapter.train()
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                x = batch_x.float().to(device)
                y = batch_y.float().to(device)
                batch = forward_mismatch_batch(args, autoencoder, latenttsf, adapter, x, y)

                pred_loss = F.mse_loss(batch["y_hat"], batch["y_true"])
                reg_loss = (batch["y_hat_full"] - batch["y_base_full"]).pow(2).mean()
                corr_loss = F.mse_loss(batch["corr_hat"], batch["corr_x"].detach())
                loss = pred_loss + args.lambda_reg * reg_loss + corr_weight * corr_loss

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
                optimizer.step()

        train_metrics = evaluate_loader(args, adapter, autoencoder, latenttsf, train_data, eval_train_loader, device)
        val_metrics = evaluate_loader(args, adapter, autoencoder, latenttsf, val_data, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch {epoch:03d} | "
            f"train base MSE {train_metrics['base_mse']:.6f} | "
            f"train adapter MSE {train_metrics['adapter_mse']:.6f} | "
            f"train gain {train_metrics['gain_mse']:.6f} | "
            f"vali base MSE {val_metrics['base_mse']:.6f} | "
            f"vali adapter MSE {val_metrics['adapter_mse']:.6f} | "
            f"vali gain {val_metrics['gain_mse']:.6f}",
            flush=True,
        )
        print(
            f"           gate mean/std {val_metrics['gate_mean']:.6f}/{val_metrics['gate_std']:.6f} | "
            f"corr_loss {val_metrics['corr_loss']:.6f} | "
            f"corr_y_delta {val_metrics['correction_abs_mean']:.6f} | "
            f"rel_corr_y_delta {val_metrics['relative_correction']:.6f}",
            flush=True,
        )

        if val_metrics["adapter_mse"] < best_val_mse:
            best_val_mse = val_metrics["adapter_mse"]
            best_val_base_mse = val_metrics["base_mse"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
            torch.save(
                {
                    "adapter_state_dict": best_state,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_adapter_mse": best_val_mse,
                    "val_base_mse": best_val_base_mse,
                },
                os.path.join(args.output_dir, "best_corr_mismatch_adapter.pt"),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        adapter.load_state_dict(best_state)

    train_metrics = evaluate_loader(args, adapter, autoencoder, latenttsf, train_data, eval_train_loader, device)
    val_metrics = evaluate_loader(args, adapter, autoencoder, latenttsf, val_data, val_loader, device)
    test_metrics = evaluate_loader(args, adapter, autoencoder, latenttsf, test_data, test_loader, device)

    baseline_valid = True
    baseline_warning = None
    if expected_metrics is not None:
        mse_gap = abs(test_metrics["base_mse"] - expected_metrics["mse"])
        if mse_gap > args.baseline_tol:
            baseline_valid = False
            baseline_warning = (
                f"Baseline mismatch: reproduced test MSE={test_metrics['base_mse']:.8f} "
                f"but expected {expected_metrics['mse']:.8f} from {expected_metrics['source']} "
                f"(gap={mse_gap:.8f}). Adapter gain is invalid."
            )
            print(f"WARNING: {baseline_warning}", flush=True)
        elif expected_metrics["mae"] is not None:
            mae_gap = abs(test_metrics["base_mae"] - expected_metrics["mae"])
            if mae_gap > args.baseline_tol:
                baseline_valid = False
                baseline_warning = (
                    f"Baseline mismatch: reproduced test MAE={test_metrics['base_mae']:.8f} "
                    f"but expected {expected_metrics['mae']:.8f} from {expected_metrics['source']} "
                    f"(gap={mae_gap:.8f}). Adapter gain is invalid."
                )
                print(f"WARNING: {baseline_warning}", flush=True)
    else:
        baseline_valid = False
        baseline_warning = (
            "No reference baseline metric found. "
            "Pass --expected_test_mse/--expected_test_mae or keep a companion metrics.json next to the checkpoint."
        )
        print(f"WARNING: {baseline_warning}", flush=True)

    summary = {
        "best_epoch": best_epoch,
        "best_val_adapter_mse": best_val_mse,
        "best_val_base_mse": best_val_base_mse,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "baseline_valid": baseline_valid,
        "baseline_warning": baseline_warning,
        "expected_metrics": expected_metrics,
    }
    with open(os.path.join(args.output_dir, "corr_mismatch_adapter_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o))

    print("\nCorrelation-Mismatch Calibration Adapter [final]", flush=True)
    print_split_metrics("train", train_metrics)
    print_split_metrics("vali ", val_metrics)
    print_split_metrics("test ", test_metrics)
    if baseline_warning is None:
        print("baseline reproduction check: passed", flush=True)
    else:
        print("baseline reproduction check: failed", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Correlation-mismatch calibration adapter on top of a frozen LatentTSF baseline"
    )
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/latenttsf_corr_mismatch_adapter")
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
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_corr", type=float, default=0.01)
    parser.add_argument("--corr_eps", type=float, default=1e-6)
    parser.add_argument("--adapter_mode", type=str, default="full", choices=ABLATION_CHOICES)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--expected_test_mse", type=float, default=None)
    parser.add_argument("--expected_test_mae", type=float, default=None)
    parser.add_argument("--baseline_tol", type=float, default=1e-5)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
