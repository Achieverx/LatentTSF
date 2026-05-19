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
    freeze_module,
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


class CandidateGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_candidates = args.M
        self.d_model = args.d_model
        hidden_dim = args.d_model * 4
        in_dim = args.d_model * 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(args.dropout),
        )
        self.out_proj = nn.Conv1d(hidden_dim, args.M * args.d_model, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Tiny candidate-specific offsets break symmetry while keeping z_cands close to z_base at init.
        self.candidate_offsets = nn.Parameter(1e-3 * torch.randn(1, args.M, 1, args.d_model))

    def forward(self, z_x, z_base):
        context = z_x.mean(dim=1, keepdim=True).expand(-1, z_base.size(1), -1)
        inp = torch.cat([z_base, context], dim=-1)
        hidden = self.net(inp.transpose(1, 2))
        delta = self.out_proj(hidden)
        bsz, _, horizon = delta.shape
        delta = delta.transpose(1, 2).reshape(bsz, horizon, self.num_candidates, -1).permute(0, 2, 1, 3)
        return delta + self.candidate_offsets


class FusionGate(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_dim = args.d_model * 4
        input_dim = args.d_model * 4
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_x, z_base, z_cands):
        cand_mean = z_cands.mean(dim=2)
        cand_res = (z_cands - z_base[:, None, :, :]).mean(dim=2)
        base_ctx = torch.cat([z_base.mean(dim=1), z_x.mean(dim=1)], dim=-1)
        base_ctx = base_ctx[:, None, :].expand(-1, z_cands.size(1), -1)
        gate_inp = torch.cat([cand_mean, cand_res, base_ctx], dim=-1)
        scores = self.net(gate_inp).squeeze(-1)
        return scores


def mse_per_sample(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean(dim=(1, 2))


def mse_per_sample_per_candidate(y_cands, y_true):
    return ((y_cands - y_true[:, None, :, :]) ** 2).mean(dim=(2, 3))


def latent_mse_per_sample_per_candidate(z_cands, z_true):
    return ((z_cands - z_true[:, None, :, :]) ** 2).mean(dim=(2, 3))


def pairwise_diversity(z_cands):
    if z_cands.size(1) <= 1:
        zero = z_cands.new_zeros(())
        return zero, zero
    z_flat = z_cands.mean(dim=2)
    pairwise = torch.cdist(z_flat, z_flat, p=2)
    mask = torch.triu(torch.ones_like(pairwise[0], dtype=torch.bool), diagonal=1)
    values = pairwise[:, mask]
    diversity = values.mean()
    loss_div = -diversity
    return diversity, loss_div


def block_values_from_sqerr(sqerr, block_ranges):
    values = []
    for start, end in block_ranges:
        values.append(float(sqerr[:, start:end, :].mean().item()))
    return values


def aggregate_block_sums(target, values, batch_size):
    for idx, value in enumerate(values):
        target[idx] += float(value) * batch_size


def premise_random_oracle(args, autoencoder, base_model, loader, device, block_ranges):
    base_sum = 0.0
    oracle_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)
            z_x = autoencoder.encode(x)
            z_base = base_model(z_x)
            y_base = autoencoder.decode(z_base)
            base_mse = mse_per_sample(y_base, y)
            best_mse = base_mse.clone()

            num_done = 0
            while num_done < args.premise_random_samples:
                chunk = min(args.premise_chunk, args.premise_random_samples - num_done)
                noise = torch.randn(
                    z_base.size(0),
                    chunk,
                    z_base.size(1),
                    z_base.size(2),
                    device=device,
                    dtype=z_base.dtype,
                )
                z_cand = z_base[:, None, :, :] + args.beta * torch.tanh(noise)
                y_cand = autoencoder.decode(z_cand.reshape(-1, z_base.size(1), z_base.size(2))).reshape(
                    z_base.size(0), chunk, z_base.size(1), y.size(-1)
                )
                cand_mse = ((y_cand - y[:, None, :, :]) ** 2).mean(dim=(2, 3))
                best_mse = torch.minimum(best_mse, cand_mse.min(dim=1).values)
                num_done += chunk

            bsz = x.size(0)
            base_sum += base_mse.sum().item()
            oracle_sum += best_mse.sum().item()
            count += bsz

    base_obs_mse = base_sum / max(count, 1)
    best_random_obs_mse = oracle_sum / max(count, 1)
    return {
        "base_obs_mse": base_obs_mse,
        "best_random_obs_mse": best_random_obs_mse,
        "oracle_obs_gain": base_obs_mse - best_random_obs_mse,
    }


def build_candidates(args, autoencoder, candidate_generator, base_model, batch_x, batch_y, device):
    x = batch_x.float().to(device)
    y = batch_y[:, -args.pred_len :, :].float().to(device)
    with torch.no_grad():
        z_x = autoencoder.encode(x)
        z_y = autoencoder.encode(y)
        z_base = base_model(z_x)
        y_base = autoencoder.decode(z_base)

    delta = candidate_generator(z_x, z_base)
    z_cands = z_base[:, None, :, :] + args.beta * torch.tanh(delta)
    y_cands = autoencoder.decode(z_cands.reshape(-1, z_base.size(1), z_base.size(2))).reshape(
        z_base.size(0), args.M, z_base.size(1), y.size(-1)
    )
    return x, y, z_x, z_y, z_base, y_base, delta, z_cands, y_cands


def apply_fusion(args, fusion_gate, z_x, z_base, z_cands):
    if args.fusion_mode == "uniform":
        weights = torch.full(
            (z_cands.size(0), z_cands.size(1)),
            1.0 / float(z_cands.size(1)),
            device=z_cands.device,
            dtype=z_cands.dtype,
        )
    else:
        scores = fusion_gate(z_x, z_base, z_cands)
        weights = torch.softmax(scores, dim=1)
    z_fused = z_base + (weights[:, :, None, None] * (z_cands - z_base[:, None, :, :])).sum(dim=1)
    return weights, z_fused


def compute_losses(args, y, z_y, z_base, y_base, delta, z_cands, y_cands, weights, z_fused, y_hat):
    dist_obs = mse_per_sample_per_candidate(y_cands, y)
    dist_lat = latent_mse_per_sample_per_candidate(z_cands, z_y)
    base_obs_mse = F.mse_loss(y_base, y)
    base_latent_mse = F.mse_loss(z_base, z_y)
    fused_obs_mse = F.mse_loss(y_hat, y)
    fused_obs_mae = F.l1_loss(y_hat, y)
    fused_latent_mse = F.mse_loss(z_fused, z_y)
    min_obs_mse = dist_obs.min(dim=1).values.mean()
    min_latent_mse = dist_lat.min(dim=1).values.mean()

    softmin_obs = y_hat.new_zeros(())
    softmin_lat = y_hat.new_zeros(())
    if not args.no_obs_set:
        softmin_obs = (
            -args.tau
            * (torch.logsumexp(-dist_obs / args.tau, dim=1) - math.log(dist_obs.size(1)))
        ).mean()
    if not args.no_latent_set:
        softmin_lat = (
            -args.tau
            * (torch.logsumexp(-dist_lat / args.tau, dim=1) - math.log(dist_lat.size(1)))
        ).mean()
    l_set = softmin_obs + 0.5 * softmin_lat

    diversity, l_div = pairwise_diversity(z_cands)
    l_anchor = (z_cands - z_base[:, None, :, :]).pow(2).mean()
    total = (
        args.lambda_forecast * fused_obs_mse
        + args.lambda_set * l_set
        + args.lambda_div * l_div
        + args.lambda_anchor * l_anchor
    )

    delta_abs = delta.abs()
    weight_entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
    max_weight_mean = weights.max(dim=1).values.mean()
    candidate_std = z_cands.std(dim=1, unbiased=False).mean()
    anchor_dist = (z_cands - z_base[:, None, :, :]).pow(2).mean().sqrt()

    return {
        "loss_total": total,
        "base_obs_mse": base_obs_mse,
        "base_latent_mse": base_latent_mse,
        "fused_obs_mse": fused_obs_mse,
        "fused_obs_mae": fused_obs_mae,
        "fused_latent_mse": fused_latent_mse,
        "min_obs_mse": min_obs_mse,
        "min_latent_mse": min_latent_mse,
        "oracle_obs_gain": base_obs_mse - min_obs_mse,
        "fused_obs_gain": base_obs_mse - fused_obs_mse,
        "softmin_obs_loss": softmin_obs,
        "softmin_lat_loss": softmin_lat,
        "loss_set": l_set,
        "loss_div": l_div,
        "pairwise_diversity": diversity,
        "loss_anchor": l_anchor,
        "delta_abs_mean": delta_abs.mean(),
        "delta_abs_max": delta_abs.max(),
        "candidate_std_across_M": candidate_std,
        "anchor_dist": anchor_dist,
        "weight_entropy": weight_entropy,
        "max_weight_mean": max_weight_mean,
        "weights": weights,
        "dist_obs": dist_obs,
        "dist_lat": dist_lat,
    }


def evaluate(args, autoencoder, base_model, candidate_generator, fusion_gate, loader, device, block_ranges):
    candidate_generator.eval()
    if fusion_gate is not None:
        fusion_gate.eval()

    sums = {
        "loss_total": 0.0,
        "base_obs_mse": 0.0,
        "base_latent_mse": 0.0,
        "fused_obs_mse": 0.0,
        "fused_obs_mae": 0.0,
        "fused_latent_mse": 0.0,
        "min_obs_mse": 0.0,
        "min_latent_mse": 0.0,
        "oracle_obs_gain": 0.0,
        "fused_obs_gain": 0.0,
        "softmin_obs_loss": 0.0,
        "softmin_lat_loss": 0.0,
        "loss_set": 0.0,
        "loss_div": 0.0,
        "pairwise_diversity": 0.0,
        "loss_anchor": 0.0,
        "delta_abs_mean": 0.0,
        "delta_abs_max": 0.0,
        "candidate_std_across_M": 0.0,
        "anchor_dist": 0.0,
        "weight_entropy": 0.0,
        "max_weight_mean": 0.0,
    }
    count = 0
    block_base_obs = [0.0 for _ in block_ranges]
    block_fused_obs = [0.0 for _ in block_ranges]
    block_base_latent = [0.0 for _ in block_ranges]
    block_fused_latent = [0.0 for _ in block_ranges]
    block_gate = [0.0 for _ in block_ranges]

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            x, y, z_x, z_y, z_base, y_base, delta, z_cands, y_cands = build_candidates(
                args, autoencoder, candidate_generator, base_model, batch_x, batch_y, device
            )
            weights, z_fused = apply_fusion(args, fusion_gate, z_x, z_base, z_cands)
            y_hat = autoencoder.decode(z_fused)
            metrics = compute_losses(args, y, z_y, z_base, y_base, delta, z_cands, y_cands, weights, z_fused, y_hat)

            bsz = x.size(0)
            count += bsz
            for key in sums:
                value = metrics[key]
                sums[key] += value.item() * bsz

            base_obs_sq = (y_base - y).pow(2)
            fused_obs_sq = (y_hat - y).pow(2)
            base_lat_sq = (z_base - z_y).pow(2)
            fused_lat_sq = (z_fused - z_y).pow(2)
            aggregate_block_sums(block_base_obs, block_values_from_sqerr(base_obs_sq, block_ranges), bsz)
            aggregate_block_sums(block_fused_obs, block_values_from_sqerr(fused_obs_sq, block_ranges), bsz)
            aggregate_block_sums(block_base_latent, block_values_from_sqerr(base_lat_sq, block_ranges), bsz)
            aggregate_block_sums(block_fused_latent, block_values_from_sqerr(fused_lat_sq, block_ranges), bsz)

            for idx, (start, end) in enumerate(block_ranges):
                block_gate[idx] += float(metrics["weights"][:, :,].mean(dim=1).mean().item()) * bsz

    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["block_base_obs"] = [value / max(count, 1) for value in block_base_obs]
    result["block_fused_obs"] = [value / max(count, 1) for value in block_fused_obs]
    result["block_base_latent"] = [value / max(count, 1) for value in block_base_latent]
    result["block_fused_latent"] = [value / max(count, 1) for value in block_fused_latent]
    result["gate_mean_block"] = [value / max(count, 1) for value in block_gate]
    return result


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_checkpoint(path, args, candidate_generator, fusion_gate, epoch, premise, val_metrics, test_metrics):
    torch.save(
        {
            "args": vars(args),
            "epoch": epoch,
            "candidate_generator_state_dict": candidate_generator.state_dict(),
            "fusion_gate_state_dict": None if fusion_gate is None else fusion_gate.state_dict(),
            "premise": premise,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


def append_history(csv_path, history, row):
    history.append(row)
    pd.DataFrame(history).to_csv(csv_path, index=False)


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    base_model, checkpoint = load_forecaster_from_checkpoint(
        args,
        device,
        args.base_checkpoint,
        LatentForecaster,
        freeze=True,
        return_checkpoint=True,
    )
    base_model.eval()
    block_ranges = compute_block_ranges(args.pred_len, args.num_blocks)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    premise = premise_random_oracle(args, autoencoder, base_model, val_loader, device, block_ranges)
    premise["warning"] = premise["oracle_obs_gain"] <= 0.0
    print(
        f"Premise check | base_obs_mse={premise['base_obs_mse']:.6f} "
        f"best_random_obs_mse={premise['best_random_obs_mse']:.6f} "
        f"oracle_obs_gain={premise['oracle_obs_gain']:.6f}",
        flush=True,
    )
    if premise["warning"]:
        print("WARNING: oracle_obs_gain <= 0 on validation set.", flush=True)
        if not args.force_train:
            summary = {
                "premise": premise,
                "stopped_before_training": True,
                "reason": "oracle_obs_gain <= 0 and force_train == 0",
            }
            with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return

    candidate_generator = CandidateGenerator(args).to(device)
    fusion_gate = None if args.fusion_mode == "uniform" else FusionGate(args).to(device)
    trainable_params = list(candidate_generator.parameters()) + ([] if fusion_gate is None else list(fusion_gate.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"Latent Candidates | model={args.model} data={args.data} pred_len={args.pred_len} "
        f"M={args.M} beta={args.beta} tau={args.tau} fusion={args.fusion_mode}",
        flush=True,
    )
    print(f"Base checkpoint: {args.base_checkpoint}", flush=True)
    print(f"Trainable params | generator: {count_params(candidate_generator)} | gate: {0 if fusion_gate is None else count_params(fusion_gate)}", flush=True)

    history = []
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    best_path = os.path.join(args.output_dir, "best_checkpoint.pth")
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_val_metrics = None
    best_test_metrics = None

    for epoch in range(1, args.epochs + 1):
        candidate_generator.train()
        if fusion_gate is not None:
            fusion_gate.train()
        train_sums = {
            "loss_total": 0.0,
            "base_obs_mse": 0.0,
            "base_latent_mse": 0.0,
            "fused_obs_mse": 0.0,
            "fused_obs_mae": 0.0,
            "fused_latent_mse": 0.0,
            "min_obs_mse": 0.0,
            "min_latent_mse": 0.0,
            "oracle_obs_gain": 0.0,
            "fused_obs_gain": 0.0,
            "softmin_obs_loss": 0.0,
            "softmin_lat_loss": 0.0,
            "loss_set": 0.0,
            "loss_div": 0.0,
            "pairwise_diversity": 0.0,
            "loss_anchor": 0.0,
            "delta_abs_mean": 0.0,
            "delta_abs_max": 0.0,
            "candidate_std_across_M": 0.0,
            "anchor_dist": 0.0,
            "weight_entropy": 0.0,
            "max_weight_mean": 0.0,
        }
        train_count = 0

        for batch_x, batch_y, _, _ in train_loader:
            x, y, z_x, z_y, z_base, y_base, delta, z_cands, y_cands = build_candidates(
                args, autoencoder, candidate_generator, base_model, batch_x, batch_y, device
            )
            weights, z_fused = apply_fusion(args, fusion_gate, z_x, z_base, z_cands)
            y_hat = autoencoder.decode(z_fused)
            metrics = compute_losses(args, y, z_y, z_base, y_base, delta, z_cands, y_cands, weights, z_fused, y_hat)
            optimizer.zero_grad()
            metrics["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            bsz = x.size(0)
            train_count += bsz
            for key in train_sums:
                train_sums[key] += metrics[key].item() * bsz

        train_metrics = {key: value / max(train_count, 1) for key, value in train_sums.items()}
        val_metrics = evaluate(args, autoencoder, base_model, candidate_generator, fusion_gate, val_loader, device, block_ranges)
        test_metrics = evaluate(args, autoencoder, base_model, candidate_generator, fusion_gate, test_loader, device, block_ranges)

        row = {
            "epoch": epoch,
            "train_base_obs_mse": train_metrics["base_obs_mse"],
            "train_fused_obs_mse": train_metrics["fused_obs_mse"],
            "train_fused_obs_gain": train_metrics["fused_obs_gain"],
            "val_base_obs_mse": val_metrics["base_obs_mse"],
            "val_fused_obs_mse": val_metrics["fused_obs_mse"],
            "val_fused_obs_gain": val_metrics["fused_obs_gain"],
            "val_oracle_obs_gain": val_metrics["oracle_obs_gain"],
            "val_loss_total": val_metrics["loss_total"],
            "test_base_obs_mse": test_metrics["base_obs_mse"],
            "test_fused_obs_mse": test_metrics["fused_obs_mse"],
            "test_fused_obs_gain": test_metrics["fused_obs_gain"],
            "test_oracle_obs_gain": test_metrics["oracle_obs_gain"],
            "test_loss_total": test_metrics["loss_total"],
        }
        append_history(csv_path, history, row)

        print(
            f"epoch {epoch:03d} | train base/fused {train_metrics['base_obs_mse']:.6f}->{train_metrics['fused_obs_mse']:.6f} "
            f"(gain {train_metrics['fused_obs_gain']:.6f}) | "
            f"val base/fused {val_metrics['base_obs_mse']:.6f}->{val_metrics['fused_obs_mse']:.6f} "
            f"(gain {val_metrics['fused_obs_gain']:.6f}) oracle {val_metrics['oracle_obs_gain']:.6f} | "
            f"test base/fused {test_metrics['base_obs_mse']:.6f}->{test_metrics['fused_obs_mse']:.6f} "
            f"(gain {test_metrics['fused_obs_gain']:.6f}) oracle {test_metrics['oracle_obs_gain']:.6f}",
            flush=True,
        )
        print(
            f"           softmin obs/lat {val_metrics['softmin_obs_loss']:.6f}/{val_metrics['softmin_lat_loss']:.6f} | "
            f"div {val_metrics['pairwise_diversity']:.6f} | anchor {val_metrics['loss_anchor']:.6f} | "
            f"entropy {val_metrics['weight_entropy']:.6f} | maxw {val_metrics['max_weight_mean']:.6f}",
            flush=True,
        )
        print(
            f"           block base obs {format_block_values(val_metrics['block_base_obs'])} | "
            f"block fused obs {format_block_values(val_metrics['block_fused_obs'])}",
            flush=True,
        )
        print(
            f"           block base lat {format_block_values(val_metrics['block_base_latent'])} | "
            f"block fused lat {format_block_values(val_metrics['block_fused_latent'])}",
            flush=True,
        )

        if val_metrics["fused_obs_mse"] < best_val:
            best_val = val_metrics["fused_obs_mse"]
            best_epoch = epoch
            bad_epochs = 0
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            save_checkpoint(best_path, args, candidate_generator, fusion_gate, epoch, premise, val_metrics, test_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    summary = {
        "premise": premise,
        "best_epoch": best_epoch,
        "best_val_metric": best_val,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "history": history,
    }
    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if best_test_metrics is not None:
        with open(os.path.join(args.output_dir, "test_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(best_test_metrics, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description="Decoder-compatible future-separable latent candidates")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
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
    parser.add_argument("--model", type=str, default="DLinear")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--fusion_mode", type=str, default="learned", choices=["uniform", "learned"])

    parser.add_argument("--lambda_forecast", type=float, default=1.0)
    parser.add_argument("--lambda_set", type=float, default=0.3)
    parser.add_argument("--lambda_div", type=float, default=0.005)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)

    parser.add_argument("--no_obs_set", type=int, default=0)
    parser.add_argument("--no_latent_set", type=int, default=0)
    parser.add_argument("--force_train", type=int, default=0)

    parser.add_argument("--premise_random_samples", type=int, default=200)
    parser.add_argument("--premise_chunk", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    args.no_obs_set = bool(args.no_obs_set)
    args.no_latent_set = bool(args.no_latent_set)
    args.force_train = bool(args.force_train)
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
