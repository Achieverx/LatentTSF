import argparse
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


class TemporalConvContextEncoder(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, z_x):
        pooled = self.conv(z_x.transpose(1, 2)).squeeze(-1)
        return self.proj(pooled)


class LatentKPM_MEMTS(nn.Module):
    def __init__(self, pred_len, d_model, K=4, hidden=256, alpha=0.02):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.K = K
        self.hidden = hidden
        self.alpha = alpha

        self.context_encoder = TemporalConvContextEncoder(d_model, hidden)
        self.branch_queries = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.horizon_queries = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)

        self.future_decoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.fusion_net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, K),
        )
        self.rho_net = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_x, z_base):
        context = self.context_encoder(z_x)
        tokens = (
            context[:, None, None, :]
            + self.branch_queries[None, :, None, :]
            + self.horizon_queries[None, None, :, :]
        )
        delta = torch.tanh(self.future_decoder(tokens))
        z_k = z_base[:, None, :, :] + self.alpha * delta

        summary = torch.cat([z_x.mean(dim=1), z_base.mean(dim=1)], dim=-1)
        pi = torch.softmax(self.fusion_net(summary), dim=-1)
        rho = self.rho_net(summary).view(z_base.size(0), 1, 1)

        z_mem = torch.einsum("bk,bkhd->bhd", pi, z_k)
        z_hat = z_base + rho * (z_mem - z_base)

        diagnostics = {
            "pi_entropy": -(pi * torch.log(pi + 1e-9)).sum(dim=-1).mean(),
            "pi_max_mean": pi.max(dim=-1).values.mean(),
            "rho_mean": rho.mean(),
            "rho_max": rho.max(),
            "rho_min": rho.min(),
            "delta_abs_mean": delta.abs().mean(),
            "delta_abs_max": delta.abs().max(),
        }
        return z_hat, z_k, delta, pi, rho, diagnostics


def off_diag_mean(matrix):
    k = matrix.size(-1)
    if k <= 1:
        return matrix.new_zeros(())
    mask = ~torch.eye(k, dtype=torch.bool, device=matrix.device)
    return matrix[..., mask].mean()


def branch_diversity_loss(z_k, z_base):
    corr = z_k - z_base[:, None]
    corr_flat = corr.reshape(corr.size(0), corr.size(1), -1)
    corr_norm = F.normalize(corr_flat, dim=-1, eps=1e-6)
    sim = torch.matmul(corr_norm, corr_norm.transpose(1, 2))
    sim_sq = sim.pow(2)
    return off_diag_mean(sim_sq)


def branch_diversity_metric(z_k, z_base):
    corr = z_k - z_base[:, None]
    corr_flat = corr.reshape(corr.size(0), corr.size(1), -1)
    corr_norm = F.normalize(corr_flat, dim=-1, eps=1e-6)
    sim = torch.matmul(corr_norm, corr_norm.transpose(1, 2))
    return off_diag_mean(sim)


def branch_oracle_entropy(best_idx, K):
    hist = torch.bincount(best_idx, minlength=K).float()
    probs = hist / hist.sum().clamp_min(1.0)
    entropy = -(probs * torch.log(probs + 1e-9)).sum()
    return entropy, hist, probs


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


def mean_block_mses(pred, target, block_ranges):
    sq = (pred - target).pow(2)
    return [float(sq[:, start:end, :].mean().item()) for start, end in block_ranges]


def compute_losses(args, y_hat, y_k, y, z_hat, z_k, z_base, rho):
    loss_final = F.mse_loss(y_hat, y)
    e_k = (y_k - y[:, None]).pow(2).mean(dim=(2, 3))
    loss_kpm = (-args.tau * torch.logsumexp(-e_k / args.tau, dim=1)).mean()
    loss_anchor = F.mse_loss(z_hat, z_base.detach())
    loss_branch_anchor = F.mse_loss(z_k, z_base[:, None].detach())
    loss_div = branch_diversity_loss(z_k, z_base)
    loss_rho = rho.mean()
    loss_total = (
        loss_final
        + args.lambda_kpm * loss_kpm
        + args.lambda_anchor * loss_anchor
        + args.lambda_branch_anchor * loss_branch_anchor
        + args.lambda_div * loss_div
        + args.lambda_rho * loss_rho
    )
    return {
        "loss_final": loss_final,
        "loss_kpm": loss_kpm,
        "loss_anchor": loss_anchor,
        "loss_branch_anchor": loss_branch_anchor,
        "loss_div": loss_div,
        "loss_rho": loss_rho,
        "loss_total": loss_total,
        "e_k": e_k,
    }


def evaluate(args, autoencoder, base_model, kpm, loader, device, block_ranges):
    kpm.eval()
    sums = {
        "base_obs_mse": 0.0,
        "base_mae": 0.0,
        "final_obs_mse": 0.0,
        "final_mae": 0.0,
        "final_obs_gain": 0.0,
        "branch_oracle_mse": 0.0,
        "branch_oracle_gain": 0.0,
        "branch_mean_mse": 0.0,
        "pi_entropy": 0.0,
        "pi_max_mean": 0.0,
        "rho_mean": 0.0,
        "rho_max": 0.0,
        "rho_min": 0.0,
        "anchor_dist": 0.0,
        "branch_anchor_dist": 0.0,
        "delta_abs_mean": 0.0,
        "delta_abs_max": 0.0,
        "branch_diversity_cos": 0.0,
        "loss_final": 0.0,
        "loss_kpm": 0.0,
        "loss_anchor": 0.0,
        "loss_branch_anchor": 0.0,
        "loss_div": 0.0,
        "loss_rho": 0.0,
        "loss_total": 0.0,
    }
    count = 0
    block_base = [0.0 for _ in block_ranges]
    block_final = [0.0 for _ in block_ranges]
    block_gain = [0.0 for _ in block_ranges]
    block_oracle = [0.0 for _ in block_ranges]
    block_oracle_gain = [0.0 for _ in block_ranges]
    pi_sum = None
    best_branch_hist = torch.zeros(args.K, dtype=torch.float32)

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            z_x = autoencoder.encode(x)
            z_base = base_model(z_x)
            y_base = autoencoder.decode(z_base)

            z_hat, z_k, delta, pi, rho, diag = kpm(z_x.detach(), z_base.detach())
            bsz, k, horizon, dim = z_k.shape
            y_k = autoencoder.decode(z_k.reshape(bsz * k, horizon, dim)).reshape(bsz, k, horizon, -1)
            y_hat = autoencoder.decode(z_hat)

            losses = compute_losses(args, y_hat, y_k, y, z_hat, z_k, z_base, rho)
            e_k = losses["e_k"]
            branch_oracle = e_k.min(dim=1)
            branch_mean = e_k.mean(dim=1)

            base_obs_mse = F.mse_loss(y_base, y)
            base_mae = F.l1_loss(y_base, y)
            final_obs_mse = F.mse_loss(y_hat, y)
            final_mae = F.l1_loss(y_hat, y)
            final_obs_gain = base_obs_mse - final_obs_mse
            branch_oracle_mse = branch_oracle.values.mean()
            branch_oracle_gain = base_obs_mse - branch_oracle_mse
            branch_mean_mse = branch_mean.mean()
            anchor_dist = F.mse_loss(z_hat, z_base.detach())
            branch_anchor_dist = F.mse_loss(z_k, z_base[:, None].detach())
            branch_div = branch_diversity_metric(z_k, z_base)

            count += bsz
            for key, value in [
                ("base_obs_mse", base_obs_mse.item()),
                ("base_mae", base_mae.item()),
                ("final_obs_mse", final_obs_mse.item()),
                ("final_mae", final_mae.item()),
                ("final_obs_gain", final_obs_gain.item()),
                ("branch_oracle_mse", branch_oracle_mse.item()),
                ("branch_oracle_gain", branch_oracle_gain.item()),
                ("branch_mean_mse", branch_mean_mse.item()),
                ("pi_entropy", diag["pi_entropy"].item()),
                ("pi_max_mean", diag["pi_max_mean"].item()),
                ("rho_mean", diag["rho_mean"].item()),
                ("rho_max", diag["rho_max"].item()),
                ("rho_min", diag["rho_min"].item()),
                ("anchor_dist", anchor_dist.item()),
                ("branch_anchor_dist", branch_anchor_dist.item()),
                ("delta_abs_mean", diag["delta_abs_mean"].item()),
                ("delta_abs_max", diag["delta_abs_max"].item()),
                ("branch_diversity_cos", branch_div.item()),
                ("loss_final", losses["loss_final"].item()),
                ("loss_kpm", losses["loss_kpm"].item()),
                ("loss_anchor", losses["loss_anchor"].item()),
                ("loss_branch_anchor", losses["loss_branch_anchor"].item()),
                ("loss_div", losses["loss_div"].item()),
                ("loss_rho", losses["loss_rho"].item()),
                ("loss_total", losses["loss_total"].item()),
            ]:
                sums[key] += value * bsz

            if pi_sum is None:
                pi_sum = pi.detach().sum(dim=0)
            else:
                pi_sum += pi.detach().sum(dim=0)

            oracle_entropy, hist, _ = branch_oracle_entropy(branch_oracle.indices, args.K)
            best_branch_hist += hist.cpu()
            sums.setdefault("branch_usage_entropy_oracle", 0.0)
            sums["branch_usage_entropy_oracle"] += oracle_entropy.item() * bsz

            base_blocks = mean_block_mses(y_base, y, block_ranges)
            final_blocks = mean_block_mses(y_hat, y, block_ranges)
            for idx, (start, end) in enumerate(block_ranges):
                branch_block_sq = (y_k[:, :, start:end, :] - y[:, None, start:end, :]).pow(2).mean(dim=(2, 3))
                oracle_block_mse = branch_block_sq.min(dim=1).values.mean().item()
                block_base[idx] += base_blocks[idx] * bsz
                block_final[idx] += final_blocks[idx] * bsz
                block_gain[idx] += (base_blocks[idx] - final_blocks[idx]) * bsz
                block_oracle[idx] += oracle_block_mse * bsz
                block_oracle_gain[idx] += (base_blocks[idx] - oracle_block_mse) * bsz

    result = {key: value / max(count, 1) for key, value in sums.items()}
    pi_soft = (pi_sum / max(count, 1)).cpu()
    result["pi_soft"] = pi_soft.tolist()
    result["pi_soft_min"] = float(pi_soft.min().item())
    result["pi_soft_max"] = float(pi_soft.max().item())
    result["pi_soft_std"] = float(pi_soft.std(unbiased=False).item())
    result["best_branch_id_hist"] = best_branch_hist.tolist()
    result["block_base_mse"] = [value / max(count, 1) for value in block_base]
    result["block_final_mse"] = [value / max(count, 1) for value in block_final]
    result["block_final_gain"] = [value / max(count, 1) for value in block_gain]
    result["block_oracle_mse"] = [value / max(count, 1) for value in block_oracle]
    result["block_oracle_gain"] = [value / max(count, 1) for value in block_oracle_gain]
    return result


def epoch_row(epoch, split, metrics, num_blocks):
    row = {
        "epoch": epoch,
        "split": split,
        "base_obs_mse": metrics["base_obs_mse"],
        "base_mae": metrics["base_mae"],
        "final_obs_mse": metrics["final_obs_mse"],
        "final_mae": metrics["final_mae"],
        "final_obs_gain": metrics["final_obs_gain"],
        "branch_oracle_mse": metrics["branch_oracle_mse"],
        "branch_oracle_gain": metrics["branch_oracle_gain"],
        "branch_mean_mse": metrics["branch_mean_mse"],
        "branch_usage_entropy_oracle": metrics["branch_usage_entropy_oracle"],
        "pi_entropy": metrics["pi_entropy"],
        "pi_max_mean": metrics["pi_max_mean"],
        "pi_soft_min": metrics["pi_soft_min"],
        "pi_soft_max": metrics["pi_soft_max"],
        "pi_soft_std": metrics["pi_soft_std"],
        "rho_mean": metrics["rho_mean"],
        "rho_max": metrics["rho_max"],
        "rho_min": metrics["rho_min"],
        "anchor_dist": metrics["anchor_dist"],
        "branch_anchor_dist": metrics["branch_anchor_dist"],
        "delta_abs_mean": metrics["delta_abs_mean"],
        "delta_abs_max": metrics["delta_abs_max"],
        "branch_diversity_cos": metrics["branch_diversity_cos"],
        "loss_final": metrics["loss_final"],
        "loss_kpm": metrics["loss_kpm"],
        "loss_anchor": metrics["loss_anchor"],
        "loss_branch_anchor": metrics["loss_branch_anchor"],
        "loss_div": metrics["loss_div"],
        "loss_rho": metrics["loss_rho"],
        "loss_total": metrics["loss_total"],
    }
    for idx in range(num_blocks):
        row[f"block_base_mse_b{idx}"] = metrics["block_base_mse"][idx]
        row[f"block_final_mse_b{idx}"] = metrics["block_final_mse"][idx]
        row[f"block_final_gain_b{idx}"] = metrics["block_final_gain"][idx]
        row[f"block_oracle_mse_b{idx}"] = metrics["block_oracle_mse"][idx]
        row[f"block_oracle_gain_b{idx}"] = metrics["block_oracle_gain"][idx]
    for idx, value in enumerate(metrics["best_branch_id_hist"]):
        row[f"best_branch_hist_b{idx}"] = value
    return row


def save_checkpoint(path, args, kpm, epoch, val_metrics, test_metrics):
    torch.save(
        {
            "args": vars(args),
            "epoch": epoch,
            "kpm_state_dict": kpm.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        path,
    )


def print_epoch_line(epoch, train_metrics, val_metrics, test_metrics):
    print(
        f"epoch {epoch:03d} | "
        f"train {train_metrics['base_obs_mse']:.6f}->{train_metrics['final_obs_mse']:.6f} (gain {train_metrics['final_obs_gain']:.6f}) | "
        f"val {val_metrics['base_obs_mse']:.6f}->{val_metrics['final_obs_mse']:.6f} (gain {val_metrics['final_obs_gain']:.6f}) | "
        f"test {test_metrics['base_obs_mse']:.6f}->{test_metrics['final_obs_mse']:.6f} (gain {test_metrics['final_obs_gain']:.6f})"
    )
    print(
        f"           oracle val/test {val_metrics['branch_oracle_mse']:.6f}/{test_metrics['branch_oracle_mse']:.6f} | "
        f"oracle gain {val_metrics['branch_oracle_gain']:.6f}/{test_metrics['branch_oracle_gain']:.6f}"
    )
    print(
        f"           pi ent/max {val_metrics['pi_entropy']:.6f}/{val_metrics['pi_max_mean']:.6f} | "
        f"pi soft min/max/std {val_metrics['pi_soft_min']:.6f}/{val_metrics['pi_soft_max']:.6f}/{val_metrics['pi_soft_std']:.6f}"
    )
    print(
        f"           rho mean/min/max {val_metrics['rho_mean']:.6f}/{val_metrics['rho_min']:.6f}/{val_metrics['rho_max']:.6f} | "
        f"anchor {val_metrics['anchor_dist']:.6f} | branch anchor {val_metrics['branch_anchor_dist']:.6f}"
    )
    print(
        f"           delta mean/max {val_metrics['delta_abs_mean']:.6f}/{val_metrics['delta_abs_max']:.6f} | "
        f"branch div cos {val_metrics['branch_diversity_cos']:.6f}"
    )
    print(
        f"           block final gain {format_block_values(val_metrics['block_final_gain'])} | "
        f"block oracle gain {format_block_values(val_metrics['block_oracle_gain'])}"
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

    kpm = LatentKPM_MEMTS(
        pred_len=args.pred_len,
        d_model=args.d_model,
        K=args.K,
        hidden=args.hidden,
        alpha=args.alpha,
    ).to(device)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)
    block_ranges = compute_block_ranges(args.pred_len, args.num_blocks)
    optimizer = torch.optim.AdamW(kpm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    frozen_params = count_params(autoencoder) + count_params(base_model)
    trainable_params = count_trainable_params(kpm)
    print(
        f"LatentKPM-MEMTS | model={args.model} data={args.data} pred_len={args.pred_len} "
        f"K={args.K} alpha={args.alpha} hidden={args.hidden} tau={args.tau}"
    )
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Trainable params: {trainable_params} | Frozen params: {frozen_params}")
    print(
        f"Hyperparams | lambda_kpm={args.lambda_kpm} lambda_anchor={args.lambda_anchor} "
        f"lambda_branch_anchor={args.lambda_branch_anchor} lambda_div={args.lambda_div} lambda_rho={args.lambda_rho}"
    )

    history_rows = []
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")

    for epoch in range(1, args.epochs + 1):
        kpm.train()
        for batch_x, batch_y, _, _ in train_loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            with torch.no_grad():
                z_x = autoencoder.encode(x)
                z_base = base_model(z_x)

            z_hat, z_k, _, _, rho, _ = kpm(z_x.detach(), z_base.detach())
            bsz, k, horizon, dim = z_k.shape
            y_k = autoencoder.decode(z_k.reshape(bsz * k, horizon, dim)).reshape(bsz, k, horizon, -1)
            y_hat = autoencoder.decode(z_hat)
            losses = compute_losses(args, y_hat, y_k, y, z_hat, z_k, z_base, rho)

            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(kpm.parameters(), args.grad_clip)
            optimizer.step()

        train_metrics = evaluate(args, autoencoder, base_model, kpm, eval_train_loader, device, block_ranges)
        val_metrics = evaluate(args, autoencoder, base_model, kpm, val_loader, device, block_ranges)
        test_metrics = evaluate(args, autoencoder, base_model, kpm, test_loader, device, block_ranges)
        print_epoch_line(epoch, train_metrics, val_metrics, test_metrics)

        history_rows.append(epoch_row(epoch, "train", train_metrics, len(block_ranges)))
        history_rows.append(epoch_row(epoch, "val", val_metrics, len(block_ranges)))
        history_rows.append(epoch_row(epoch, "test", test_metrics, len(block_ranges)))
        pd.DataFrame(history_rows).to_csv(csv_path, index=False)

        current_val = val_metrics["final_obs_mse"]
        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_checkpoint_path, args, kpm, epoch, val_metrics, test_metrics)
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
            break

    checkpoint_data = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    kpm.load_state_dict(checkpoint_data["kpm_state_dict"])
    kpm.eval()

    final_train = evaluate(args, autoencoder, base_model, kpm, eval_train_loader, device, block_ranges)
    final_val = evaluate(args, autoencoder, base_model, kpm, val_loader, device, block_ranges)
    final_test = evaluate(args, autoencoder, base_model, kpm, test_loader, device, block_ranges)

    summary = {
        "best_epoch": best_epoch,
        "test_base_obs_mse": final_test["base_obs_mse"],
        "test_final_obs_mse": final_test["final_obs_mse"],
        "test_final_obs_gain": final_test["final_obs_gain"],
        "test_branch_oracle_mse": final_test["branch_oracle_mse"],
        "test_branch_oracle_gain": final_test["branch_oracle_gain"],
        "test_pi_entropy": final_test["pi_entropy"],
        "test_pi_max_mean": final_test["pi_max_mean"],
        "test_rho_mean": final_test["rho_mean"],
        "K": args.K,
        "alpha": args.alpha,
        "tau": args.tau,
        "lambda_kpm": args.lambda_kpm,
        "lambda_anchor": args.lambda_anchor,
        "lambda_branch_anchor": args.lambda_branch_anchor,
        "lambda_div": args.lambda_div,
        "lambda_rho": args.lambda_rho,
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
        f"LatentKPM-MEMTS [test]\n"
        f"  obs MSE / MAE: {final_test['base_obs_mse']:.6f} / {final_test['base_mae']:.6f} -> "
        f"{final_test['final_obs_mse']:.6f} / {final_test['final_mae']:.6f}\n"
        f"  final gain: {final_test['final_obs_gain']:.6f}\n"
        f"  branch oracle MSE/gain: {final_test['branch_oracle_mse']:.6f} / {final_test['branch_oracle_gain']:.6f}\n"
        f"  pi ent/max {final_test['pi_entropy']:.6f}/{final_test['pi_max_mean']:.6f} | "
        f"rho mean {final_test['rho_mean']:.6f}\n"
        f"  block final gain: {format_block_values(final_test['block_final_gain'])}\n"
        f"  block oracle gain: {format_block_values(final_test['block_oracle_gain'])}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="LatentKPM-MEMTS for frozen LatentTSF")
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
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lambda_kpm", type=float, default=0.3)
    parser.add_argument("--lambda_anchor", type=float, default=0.5)
    parser.add_argument("--lambda_branch_anchor", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lambda_rho", type=float, default=0.001)
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
