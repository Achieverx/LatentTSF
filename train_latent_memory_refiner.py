import argparse
import csv
import json
import math
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


def block_means_from_sqerr(sqerr, block_ranges):
    return [float(sqerr[:, start:end, :].mean().item()) for start, end in block_ranges]


def format_block_values(values):
    return " | ".join(f"b{idx}:{value:.4f}" for idx, value in enumerate(values))


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


class TemporalConvQueryEncoder(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, z):
        out = self.net(z.transpose(1, 2))
        return out.squeeze(-1)


class FALMRRefiner(nn.Module):
    def __init__(self, pred_len, d_model, K=16, hidden=256, alpha=0.05, gate_mode="fixed_late"):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.K = K
        self.hidden = hidden
        self.alpha = alpha
        self.gate_mode = gate_mode

        self.slots = nn.Parameter(torch.randn(K, pred_len, d_model) * 0.01)
        self.keys = nn.Parameter(torch.randn(K, hidden) * 0.01)

        self.query_x = TemporalConvQueryEncoder(d_model, hidden)
        self.query_base = TemporalConvQueryEncoder(d_model, hidden)
        self.proj_q = nn.Linear(2 * hidden, hidden)

        self.horizon_gate = nn.Parameter(torch.full((1, pred_len, 1), -3.0))

    def _make_gate(self, z_base, gate_mode):
        if gate_mode == "fixed_late":
            gate = torch.zeros(1, self.pred_len, 1, device=z_base.device, dtype=z_base.dtype)
            gate[:, self.pred_len // 2 :, :] = 1.0
            return gate
        if gate_mode == "learned":
            return torch.sigmoid(self.horizon_gate).to(dtype=z_base.dtype, device=z_base.device)
        if gate_mode == "none":
            return torch.ones(1, self.pred_len, 1, device=z_base.device, dtype=z_base.dtype)
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    def forward(self, z_x, z_base, gate_mode=None):
        gate_mode = gate_mode or self.gate_mode
        q_x = self.query_x(z_x)
        q_b = self.query_base(z_base)
        q = self.proj_q(torch.cat([q_x, q_b], dim=-1))

        attn = torch.softmax(q @ self.keys.T / math.sqrt(q.size(-1)), dim=-1)
        delta = torch.einsum("bk,khd->bhd", attn, self.slots)
        delta = torch.tanh(delta)

        gate = self._make_gate(z_base, gate_mode)
        z_refined = z_base + self.alpha * gate * delta

        attn_entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1).mean()
        attn_max_mean = attn.max(dim=-1).values.mean()
        slot_usage_soft = attn.mean(dim=0)
        slot_usage_entropy = -(slot_usage_soft * torch.log(slot_usage_soft + 1e-9)).sum()
        slot_usage_hard = (attn > (1.0 / float(self.K))).float().sum(dim=0)

        diagnostics = {
            "attn_entropy": attn_entropy,
            "attn_max_mean": attn_max_mean,
            "slot_usage_soft": slot_usage_soft,
            "slot_usage_entropy": slot_usage_entropy,
            "slot_usage_hard": slot_usage_hard,
        }
        return z_refined, delta, attn, gate, diagnostics


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def pairwise_slot_diversity(slots):
    if slots.size(0) <= 1:
        zero = slots.new_zeros(())
        return zero, zero
    slot_summary = slots.mean(dim=1)
    pairwise = torch.pdist(slot_summary, p=2)
    diversity = pairwise.mean()
    return diversity, -diversity


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


def compute_losses(args, y_hat, y, z_refined, z_base, attn, refiner):
    loss_forecast = F.mse_loss(y_hat, y)
    loss_anchor = F.mse_loss(z_refined, z_base.detach())
    slot_pairwise_diversity, loss_div = pairwise_slot_diversity(refiner.slots)
    loss_slot_norm = refiner.slots.pow(2).mean()
    batch_mean_attn = attn.mean(dim=0)
    loss_usage = torch.sum(batch_mean_attn * torch.log(batch_mean_attn + 1e-9))

    loss_total = (
        loss_forecast
        + args.lambda_anchor * loss_anchor
        + args.lambda_div * loss_div
        + args.lambda_slot_norm * loss_slot_norm
        + args.lambda_usage * loss_usage
    )

    return {
        "loss_forecast": loss_forecast,
        "loss_anchor": loss_anchor,
        "loss_div": loss_div,
        "loss_slot_norm": loss_slot_norm,
        "loss_usage": loss_usage,
        "loss_total": loss_total,
        "slot_pairwise_diversity": slot_pairwise_diversity,
        "slot_norm": loss_slot_norm,
    }


def aggregate_mean(acc, key, value, batch_size):
    acc[key] += float(value) * batch_size


def gate_block_means(gate, block_ranges):
    gate_tensor = gate if isinstance(gate, torch.Tensor) else None
    if gate_tensor is None:
        return [1.0 for _ in block_ranges]
    means = []
    for start, end in block_ranges:
        means.append(float(gate_tensor[:, start:end, :].mean().item()))
    return means


def evaluate(args, autoencoder, base_model, refiner, loader, device, block_ranges):
    refiner.eval()
    sums = {
        "base_obs_mse": 0.0,
        "base_obs_mae": 0.0,
        "refined_obs_mse": 0.0,
        "refined_obs_mae": 0.0,
        "obs_gain": 0.0,
        "anchor_dist": 0.0,
        "delta_abs_mean": 0.0,
        "delta_abs_max": 0.0,
        "attn_entropy": 0.0,
        "attn_max_mean": 0.0,
        "slot_usage_entropy": 0.0,
        "slot_pairwise_diversity": 0.0,
        "slot_norm": 0.0,
        "loss_forecast": 0.0,
        "loss_anchor": 0.0,
        "loss_div": 0.0,
        "loss_slot_norm": 0.0,
        "loss_usage": 0.0,
        "loss_total": 0.0,
        "gate_mean_all": 0.0,
    }
    count = 0
    block_base = [0.0 for _ in block_ranges]
    block_refined = [0.0 for _ in block_ranges]
    block_gain = [0.0 for _ in block_ranges]
    gate_blocks = [0.0 for _ in block_ranges]
    slot_usage_soft_sum = None
    slot_usage_hard_sum = None

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            with torch.no_grad():
                z_x = autoencoder.encode(x)
                z_base = base_model(z_x)
                y_base = autoencoder.decode(z_base)

            z_refined, delta, attn, gate, diag = refiner(z_x.detach(), z_base.detach(), gate_mode=args.gate_mode)
            y_hat = autoencoder.decode(z_refined)
            losses = compute_losses(args, y_hat, y, z_refined, z_base, attn, refiner)

            base_obs_mse = F.mse_loss(y_base, y)
            base_obs_mae = F.l1_loss(y_base, y)
            refined_obs_mse = F.mse_loss(y_hat, y)
            refined_obs_mae = F.l1_loss(y_hat, y)
            obs_gain = base_obs_mse - refined_obs_mse
            anchor_dist = F.mse_loss(z_refined, z_base.detach())
            delta_abs_mean = delta.abs().mean()
            delta_abs_max = delta.abs().max()
            gate_mean_all = gate.mean() if isinstance(gate, torch.Tensor) else torch.tensor(1.0, device=device)

            bsz = x.size(0)
            count += bsz

            aggregate_mean(sums, "base_obs_mse", base_obs_mse.item(), bsz)
            aggregate_mean(sums, "base_obs_mae", base_obs_mae.item(), bsz)
            aggregate_mean(sums, "refined_obs_mse", refined_obs_mse.item(), bsz)
            aggregate_mean(sums, "refined_obs_mae", refined_obs_mae.item(), bsz)
            aggregate_mean(sums, "obs_gain", obs_gain.item(), bsz)
            aggregate_mean(sums, "anchor_dist", anchor_dist.item(), bsz)
            aggregate_mean(sums, "delta_abs_mean", delta_abs_mean.item(), bsz)
            aggregate_mean(sums, "delta_abs_max", delta_abs_max.item(), bsz)
            aggregate_mean(sums, "attn_entropy", diag["attn_entropy"].item(), bsz)
            aggregate_mean(sums, "attn_max_mean", diag["attn_max_mean"].item(), bsz)
            aggregate_mean(sums, "slot_usage_entropy", diag["slot_usage_entropy"].item(), bsz)
            aggregate_mean(sums, "slot_pairwise_diversity", losses["slot_pairwise_diversity"].item(), bsz)
            aggregate_mean(sums, "slot_norm", losses["slot_norm"].item(), bsz)
            aggregate_mean(sums, "loss_forecast", losses["loss_forecast"].item(), bsz)
            aggregate_mean(sums, "loss_anchor", losses["loss_anchor"].item(), bsz)
            aggregate_mean(sums, "loss_div", losses["loss_div"].item(), bsz)
            aggregate_mean(sums, "loss_slot_norm", losses["loss_slot_norm"].item(), bsz)
            aggregate_mean(sums, "loss_usage", losses["loss_usage"].item(), bsz)
            aggregate_mean(sums, "loss_total", losses["loss_total"].item(), bsz)
            aggregate_mean(sums, "gate_mean_all", gate_mean_all.item(), bsz)

            if slot_usage_soft_sum is None:
                slot_usage_soft_sum = diag["slot_usage_soft"].detach() * bsz
                slot_usage_hard_sum = diag["slot_usage_hard"].detach()
            else:
                slot_usage_soft_sum += diag["slot_usage_soft"].detach() * bsz
                slot_usage_hard_sum += diag["slot_usage_hard"].detach()

            base_sq = (y_base - y).pow(2)
            refined_sq = (y_hat - y).pow(2)
            base_blocks = block_means_from_sqerr(base_sq, block_ranges)
            refined_blocks = block_means_from_sqerr(refined_sq, block_ranges)
            gate_block_vals = gate_block_means(gate, block_ranges)
            for idx, (base_value, refined_value, gate_value) in enumerate(zip(base_blocks, refined_blocks, gate_block_vals)):
                block_base[idx] += base_value * bsz
                block_refined[idx] += refined_value * bsz
                block_gain[idx] += (base_value - refined_value) * bsz
                gate_blocks[idx] += gate_value * bsz

    result = {key: value / max(count, 1) for key, value in sums.items()}
    slot_usage_soft = (slot_usage_soft_sum / max(count, 1)).cpu()
    result["slot_usage_soft"] = slot_usage_soft.tolist()
    result["slot_usage_hard"] = slot_usage_hard_sum.cpu().tolist()
    result["slot_usage_soft_min"] = float(slot_usage_soft.min().item())
    result["slot_usage_soft_max"] = float(slot_usage_soft.max().item())
    result["slot_usage_soft_std"] = float(slot_usage_soft.std(unbiased=False).item())
    result["block_base_mse"] = [value / max(count, 1) for value in block_base]
    result["block_refined_mse"] = [value / max(count, 1) for value in block_refined]
    result["block_gain"] = [value / max(count, 1) for value in block_gain]
    result["gate_mean_blocks"] = [value / max(count, 1) for value in gate_blocks]
    return result


def epoch_row(epoch, split, metrics, num_blocks):
    row = {
        "epoch": epoch,
        "split": split,
        "base_obs_mse": metrics["base_obs_mse"],
        "base_obs_mae": metrics["base_obs_mae"],
        "refined_obs_mse": metrics["refined_obs_mse"],
        "refined_obs_mae": metrics["refined_obs_mae"],
        "obs_gain": metrics["obs_gain"],
        "anchor_dist": metrics["anchor_dist"],
        "delta_abs_mean": metrics["delta_abs_mean"],
        "delta_abs_max": metrics["delta_abs_max"],
        "attn_entropy": metrics["attn_entropy"],
        "attn_max_mean": metrics["attn_max_mean"],
        "slot_usage_entropy": metrics["slot_usage_entropy"],
        "slot_usage_soft_min": metrics["slot_usage_soft_min"],
        "slot_usage_soft_max": metrics["slot_usage_soft_max"],
        "slot_usage_soft_std": metrics["slot_usage_soft_std"],
        "slot_pairwise_diversity": metrics["slot_pairwise_diversity"],
        "slot_norm": metrics["slot_norm"],
        "gate_mean_all": metrics["gate_mean_all"],
        "loss_forecast": metrics["loss_forecast"],
        "loss_anchor": metrics["loss_anchor"],
        "loss_div": metrics["loss_div"],
        "loss_slot_norm": metrics["loss_slot_norm"],
        "loss_usage": metrics["loss_usage"],
        "loss_total": metrics["loss_total"],
    }
    for idx in range(num_blocks):
        row[f"gate_mean_b{idx}"] = metrics["gate_mean_blocks"][idx]
        row[f"block_base_mse_b{idx}"] = metrics["block_base_mse"][idx]
        row[f"block_refined_mse_b{idx}"] = metrics["block_refined_mse"][idx]
        row[f"block_gain_b{idx}"] = metrics["block_gain"][idx]
    return row


def save_checkpoint(path, args, refiner, epoch, val_metrics, test_metrics):
    torch.save(
        {
            "args": vars(args),
            "epoch": epoch,
            "refiner_state_dict": refiner.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


def print_epoch_line(epoch, train_metrics, val_metrics, test_metrics):
    print(
        f"epoch {epoch:03d} | "
        f"train obs {train_metrics['base_obs_mse']:.6f}->{train_metrics['refined_obs_mse']:.6f} (gain {train_metrics['obs_gain']:.6f}) | "
        f"val obs {val_metrics['base_obs_mse']:.6f}->{val_metrics['refined_obs_mse']:.6f} (gain {val_metrics['obs_gain']:.6f}) | "
        f"test obs {test_metrics['base_obs_mse']:.6f}->{test_metrics['refined_obs_mse']:.6f} (gain {test_metrics['obs_gain']:.6f})"
    )
    print(
        f"           anchor {val_metrics['anchor_dist']:.6f} | delta mean/max {val_metrics['delta_abs_mean']:.6f}/{val_metrics['delta_abs_max']:.6f} | "
        f"attn ent/max {val_metrics['attn_entropy']:.6f}/{val_metrics['attn_max_mean']:.6f} | "
        f"slot usage ent {val_metrics['slot_usage_entropy']:.6f}"
    )
    print(
        f"           slot soft min/max/std {val_metrics['slot_usage_soft_min']:.6f}/{val_metrics['slot_usage_soft_max']:.6f}/{val_metrics['slot_usage_soft_std']:.6f} | "
        f"slot div {val_metrics['slot_pairwise_diversity']:.6f} | slot norm {val_metrics['slot_norm']:.6f}"
    )
    print(
        f"           gate all {val_metrics['gate_mean_all']:.6f} | gate blocks {format_block_values(val_metrics['gate_mean_blocks'])}"
    )
    print(
        f"           block base {format_block_values(val_metrics['block_base_mse'])} | "
        f"block refined {format_block_values(val_metrics['block_refined_mse'])} | "
        f"block gain {format_block_values(val_metrics['block_gain'])}"
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

    base_model, checkpoint = load_forecaster_from_checkpoint(
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

    refiner = FALMRRefiner(
        pred_len=args.pred_len,
        d_model=args.d_model,
        K=args.K,
        hidden=args.hidden,
        alpha=args.alpha,
        gate_mode=args.gate_mode,
    ).to(device)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)
    block_ranges = compute_block_ranges(args.pred_len, args.num_blocks)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    frozen_params = count_params(autoencoder) + count_params(base_model)
    trainable_params = count_trainable_params(refiner)
    print(
        f"Latent Memory Refiner | model={args.model} data={args.data} pred_len={args.pred_len} "
        f"K={args.K} alpha={args.alpha} gate_mode={args.gate_mode} query_type={args.query_type}"
    )
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Trainable params: {trainable_params} | Frozen params: {frozen_params}")
    print(
        f"Hyperparams | lambda_anchor={args.lambda_anchor} lambda_div={args.lambda_div} "
        f"lambda_usage={args.lambda_usage} lambda_slot_norm={args.lambda_slot_norm}"
    )

    history_rows = []
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")
    best_val_metrics = None
    best_test_metrics = None

    for epoch in range(1, args.epochs + 1):
        refiner.train()
        for batch_x, batch_y, _, _ in train_loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            with torch.no_grad():
                z_x = autoencoder.encode(x)
                z_base = base_model(z_x)

            z_refined, delta, attn, gate, _ = refiner(z_x.detach(), z_base.detach(), gate_mode=args.gate_mode)
            y_hat = autoencoder.decode(z_refined)
            losses = compute_losses(args, y_hat, y, z_refined, z_base, attn, refiner)

            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            optimizer.step()

        train_metrics = evaluate(args, autoencoder, base_model, refiner, eval_train_loader, device, block_ranges)
        val_metrics = evaluate(args, autoencoder, base_model, refiner, val_loader, device, block_ranges)
        test_metrics = evaluate(args, autoencoder, base_model, refiner, test_loader, device, block_ranges)
        print_epoch_line(epoch, train_metrics, val_metrics, test_metrics)

        history_rows.append(epoch_row(epoch, "train", train_metrics, len(block_ranges)))
        history_rows.append(epoch_row(epoch, "val", val_metrics, len(block_ranges)))
        history_rows.append(epoch_row(epoch, "test", test_metrics, len(block_ranges)))
        pd.DataFrame(history_rows).to_csv(csv_path, index=False)

        current_val = val_metrics["refined_obs_mse"]
        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            bad_epochs = 0
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            save_checkpoint(best_checkpoint_path, args, refiner, epoch, val_metrics, test_metrics)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

    checkpoint_data = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    refiner.load_state_dict(checkpoint_data["refiner_state_dict"])
    refiner.eval()

    final_train = evaluate(args, autoencoder, base_model, refiner, eval_train_loader, device, block_ranges)
    final_val = evaluate(args, autoencoder, base_model, refiner, val_loader, device, block_ranges)
    final_test = evaluate(args, autoencoder, base_model, refiner, test_loader, device, block_ranges)

    summary = {
        "best_epoch": best_epoch,
        "best_val_refined_obs_mse": final_val["refined_obs_mse"],
        "best_val_obs_gain": final_val["obs_gain"],
        "test_base_obs_mse": final_test["base_obs_mse"],
        "test_refined_obs_mse": final_test["refined_obs_mse"],
        "test_obs_gain": final_test["obs_gain"],
        "test_base_obs_mae": final_test["base_obs_mae"],
        "test_refined_obs_mae": final_test["refined_obs_mae"],
        "K": args.K,
        "alpha": args.alpha,
        "gate_mode": args.gate_mode,
        "query_type": args.query_type,
        "lambda_anchor": args.lambda_anchor,
        "lambda_div": args.lambda_div,
        "lambda_slot_norm": args.lambda_slot_norm,
        "lambda_usage": args.lambda_usage,
        "hidden": args.hidden,
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    test_diag = {
        "train": detach_metrics(final_train),
        "val": detach_metrics(final_val),
        "test": detach_metrics(final_test),
        "block_ranges": block_ranges,
    }
    with open(os.path.join(args.output_dir, "test_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(test_diag, f, indent=2)

    print(
        f"Latent Memory Refiner [test]\n"
        f"  obs MSE / MAE: {final_test['base_obs_mse']:.6f} / {final_test['base_obs_mae']:.6f} -> "
        f"{final_test['refined_obs_mse']:.6f} / {final_test['refined_obs_mae']:.6f}\n"
        f"  obs gain: {final_test['obs_gain']:.6f}\n"
        f"  anchor {final_test['anchor_dist']:.6f} | delta mean/max {final_test['delta_abs_mean']:.6f}/{final_test['delta_abs_max']:.6f}\n"
        f"  attn entropy/max {final_test['attn_entropy']:.6f}/{final_test['attn_max_mean']:.6f} | "
        f"slot usage entropy {final_test['slot_usage_entropy']:.6f}\n"
        f"  gate blocks: {format_block_values(final_test['gate_mean_blocks'])}\n"
        f"  block gain: {format_block_values(final_test['block_gain'])}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Horizon-aware Forecast-aware Latent Memory Refinement")
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
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gate_mode", type=str, default="fixed_late", choices=["none", "fixed_late", "learned"])
    parser.add_argument("--query_type", type=str, default="temporal_conv")
    parser.add_argument("--lambda_anchor", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.001)
    parser.add_argument("--lambda_slot_norm", type=float, default=0.0001)
    parser.add_argument("--lambda_usage", type=float, default=0.0)
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
    if args.query_type != "temporal_conv":
        raise ValueError(f"Unsupported query_type: {args.query_type}. Only temporal_conv is implemented in V1.")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
