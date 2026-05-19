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
from utils.latent_script_utils import load_autoencoder, load_forecaster_from_checkpoint, official_loader, set_seed


CHECKPOINT_ARG_KEYS = [
    "task_name",
    "model",
    "data",
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
    "ae_type",
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
]


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
    return sum(param.numel() for param in module.parameters())


def count_trainable_params(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def count_frozen_params(module):
    return sum(param.numel() for param in module.parameters() if not param.requires_grad)


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


def resolve_device(device_arg):
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def sync_args_from_base_checkpoint(args):
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    for key in CHECKPOINT_ARG_KEYS:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    return checkpoint_args


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


class LinearLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LinearLoRA expects nn.Linear, got {type(linear)}")
        if rank <= 0:
            raise ValueError("rank must be positive for LoRA")

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x)))
        return base + self.scale * lora


def set_module_by_name(root_module, module_name, new_module):
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def list_named_linears(module):
    return [(name, child) for name, child in module.named_modules() if isinstance(child, nn.Linear)]


def match_any(name, keywords):
    return any(keyword in name for keyword in keywords)


def select_lora_target_names(backbone, model_name, target_mode):
    named_linears = list_named_linears(backbone)
    if not named_linears:
        return []

    names = [name for name, _ in named_linears]
    lowered = {name: name.lower() for name in names}

    def attention_projection_names():
        matched = []
        for name in names:
            lname = lowered[name]
            if any(token in lname for token in ["query_projection", "value_projection", "out_projection", "q_proj", "v_proj", "out_proj"]):
                matched.append(name)
        return matched

    def qv_names():
        matched = []
        for name in names:
            lname = lowered[name]
            if "embedding" in lname:
                continue
            q_match = any(token in lname for token in ["query_projection", "query", "q_proj", "w_q"])
            v_match = any(token in lname for token in ["value_projection", "value", "v_proj", "w_v"])
            if q_match or v_match:
                matched.append(name)
        return matched

    def is_head_target(name):
        return name in {"head.linear", "projection", "Linear_Seasonal", "Linear_Trend"}

    attention_names = sorted(set(attention_projection_names()))
    qv_target_names = sorted(set(qv_names()))

    if target_mode == "all_linear":
        selected = names
    elif target_mode == "qv_only":
        selected = qv_target_names if qv_target_names else attention_names
    elif target_mode in {"attention", "attention_only"}:
        selected = attention_names
    elif target_mode == "head":
        selected = [name for name in names if is_head_target(name)]
    elif target_mode == "auto":
        if model_name == "DLinear":
            selected = [name for name in names if name in {"Linear_Seasonal", "Linear_Trend"}]
            if not selected:
                selected = names
        elif model_name == "PatchTST":
            selected = [
                name
                for name in names
                if name in attention_names or is_head_target(name)
            ]
            if not selected:
                selected = names
        elif model_name == "iTransformer":
            selected = qv_target_names if qv_target_names else attention_names
        else:
            selected = names
    else:
        raise ValueError(f"Unsupported lora_target: {target_mode}")

    if not selected and target_mode != "all_linear" and not (target_mode == "auto" and model_name == "iTransformer"):
        selected = names
    return sorted(set(selected))


def inject_lora(backbone, model_name, use_lora, rank, alpha, dropout, target_mode):
    for param in backbone.parameters():
        param.requires_grad = False

    if not use_lora:
        return {"enabled": False, "target_modules": [], "num_target_modules": 0}

    target_names = select_lora_target_names(backbone, model_name, target_mode)
    for module_name, module in list_named_linears(backbone):
        if module_name not in target_names:
            continue
        set_module_by_name(backbone, module_name, LinearLoRA(module, rank=rank, alpha=alpha, dropout=dropout))

    return {
        "enabled": True,
        "target_modules": target_names,
        "num_target_modules": len(target_names),
    }


def safe_logit(prob):
    eps = 1e-6
    prob = min(max(prob, eps), 1.0 - eps)
    return torch.log(torch.tensor(prob / (1.0 - prob), dtype=torch.float32))


class ForecastCalibration(nn.Module):
    def __init__(
        self,
        pred_len,
        d_model,
        calib_type="lowrank",
        calib_rank=4,
        alpha=0.02,
        calib_gate="scalar",
        gate_hidden=64,
        gate_range=1.0,
        gate_init=0.5,
        num_blocks=4,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.calib_type = calib_type
        self.calib_rank = calib_rank
        self.alpha = alpha
        self.calib_gate = calib_gate
        self.gate_range = gate_range
        self.gate_init = gate_init
        self.num_blocks = max(1, min(num_blocks, pred_len))
        self.block_ranges = compute_block_ranges(pred_len, self.num_blocks)

        block_ids = torch.empty(pred_len, dtype=torch.long)
        for block_idx, (start, end) in enumerate(self.block_ranges):
            block_ids[start:end] = block_idx
        self.register_buffer("block_ids", block_ids, persistent=False)

        if calib_type == "none":
            self.delta_param = None
        elif calib_type == "global":
            self.delta_param = nn.Parameter(torch.zeros(pred_len, d_model))
        elif calib_type == "block":
            self.delta_param = nn.Parameter(torch.zeros(self.num_blocks, d_model))
        elif calib_type == "lowrank":
            self.delta_A = nn.Parameter(torch.randn(pred_len, calib_rank) * 0.01)
            self.delta_B = nn.Parameter(torch.randn(calib_rank, d_model) * 0.01)
        else:
            raise ValueError(f"Unsupported calib_type: {calib_type}")

        if calib_type == "none" or calib_gate == "none":
            self.gate_mlp = None
        else:
            out_dim = 1 if calib_gate == "scalar" else self.num_blocks
            self.gate_mlp = nn.Sequential(
                nn.Linear(3 * d_model, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, out_dim),
            )
            final_linear = self.gate_mlp[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        init_ratio = gate_init / max(gate_range, 1e-6)
        self.register_buffer("gate_init_bias", safe_logit(init_ratio), persistent=False)

    def get_delta(self):
        if self.calib_type == "none":
            return None
        if self.calib_type == "global":
            raw_delta = self.delta_param
        elif self.calib_type == "block":
            raw_delta = self.delta_param[self.block_ids]
        elif self.calib_type == "lowrank":
            raw_delta = self.delta_A @ self.delta_B
        else:
            raise RuntimeError("Unsupported calibration type")
        return torch.tanh(raw_delta)

    def get_gate(self, z_x, z_base, z_lora):
        batch_size = z_lora.size(0)
        device = z_lora.device
        dtype = z_lora.dtype

        if self.gate_mlp is None:
            gate_block = torch.ones(batch_size, self.num_blocks, device=device, dtype=dtype)
            gate_time = torch.ones(batch_size, self.pred_len, 1, device=device, dtype=dtype)
            gate_raw = None
        else:
            summary = torch.cat(
                [z_x.mean(dim=1), z_base.mean(dim=1), (z_lora - z_base).mean(dim=1)],
                dim=-1,
            )
            gate_raw = self.gate_mlp(summary)
            gate_base = self.gate_range * torch.sigmoid(gate_raw + self.gate_init_bias.to(device=device, dtype=dtype))
            if self.calib_gate == "scalar":
                gate_block = gate_base.repeat(1, self.num_blocks)
                gate_time = gate_base.unsqueeze(1).repeat(1, self.pred_len, 1)
            elif self.calib_gate == "block":
                gate_block = gate_base
                gate_time = gate_block[:, self.block_ids].unsqueeze(-1)
            else:
                raise RuntimeError("Unsupported calib_gate mode")

        gate_flat = gate_time.squeeze(-1)
        stats = {
            "gate_mean": gate_flat.mean(),
            "gate_std": gate_flat.std(unbiased=False),
            "gate_min": gate_flat.min(),
            "gate_max": gate_flat.max(),
        }
        for idx, (start, end) in enumerate(self.block_ranges):
            block_vals = gate_flat[:, start:end]
            stats[f"gate_mean_b{idx}"] = block_vals.mean()
            stats[f"gate_std_b{idx}"] = block_vals.std(unbiased=False)

        return gate_time, gate_block, gate_raw, stats

    def forward(self, z_lora, z_x, z_base):
        delta = self.get_delta()
        gate_time, gate_block, gate_raw, gate_stats = self.get_gate(z_x, z_base, z_lora)

        if delta is None:
            delta = z_lora.new_zeros(self.pred_len, self.d_model)
            residual = z_lora.new_zeros(z_lora.shape)
        else:
            residual = self.alpha * gate_time * delta.unsqueeze(0)

        z_cal = z_lora + residual
        diagnostics = {
            "delta_abs_mean": delta.abs().mean(),
            "delta_abs_max": delta.abs().max(),
            "delta_norm": delta.pow(2).mean(),
            "delta_smooth": (delta[1:] - delta[:-1]).pow(2).mean() if delta.size(0) > 1 else delta.new_zeros(()),
            "gate_time": gate_time,
            "gate_block": gate_block,
            "gate_raw": gate_raw,
        }
        diagnostics.update(gate_stats)
        return z_cal, delta, diagnostics


def compute_losses(args, y_hat, y, z_lora, z_base, z_cal, delta, gate_time):
    zero = z_cal.new_zeros(())
    loss_forecast = F.mse_loss(y_hat, y)
    loss_lora_anchor = F.mse_loss(z_lora, z_base.detach())

    if args.calib_type == "none":
        loss_cal_anchor = zero
        loss_cal_base_anchor = zero
        loss_norm = zero
        loss_smooth = zero
        loss_gate = zero
    else:
        loss_cal_anchor = F.mse_loss(z_cal, z_lora.detach())
        loss_cal_base_anchor = F.mse_loss(z_cal, z_base.detach())
        loss_norm = delta.pow(2).mean()
        loss_smooth = (delta[1:] - delta[:-1]).pow(2).mean() if delta.size(0) > 1 else zero
        if args.calib_gate == "none":
            loss_gate = zero
        else:
            loss_gate = (gate_time - args.gate_init).pow(2).mean()

    loss_total = (
        loss_forecast
        + args.lambda_lora_anchor * loss_lora_anchor
        + args.lambda_cal_anchor * loss_cal_anchor
        + args.lambda_cal_base_anchor * loss_cal_base_anchor
        + args.lambda_norm * loss_norm
        + args.lambda_smooth * loss_smooth
        + args.lambda_gate * loss_gate
    )
    return {
        "loss_forecast": loss_forecast,
        "loss_lora_anchor": loss_lora_anchor,
        "loss_cal_anchor": loss_cal_anchor,
        "loss_cal_base_anchor": loss_cal_base_anchor,
        "loss_norm": loss_norm,
        "loss_smooth": loss_smooth,
        "loss_gate": loss_gate,
        "loss_total": loss_total,
    }


def evaluate(args, autoencoder, base_model, adapted_model, calibrator, loader, device, block_ranges):
    autoencoder.eval()
    base_model.eval()
    adapted_model.eval()
    calibrator.eval()

    sums = {
        "base_obs_mse": 0.0,
        "adapted_obs_mse": 0.0,
        "obs_gain": 0.0,
        "base_mae": 0.0,
        "adapted_mae": 0.0,
        "lora_anchor": 0.0,
        "cal_to_lora_anchor": 0.0,
        "cal_base_anchor": 0.0,
        "relative_z_lora_change": 0.0,
        "relative_z_cal_change": 0.0,
        "relative_cal_extra_change": 0.0,
        "delta_abs_mean": 0.0,
        "delta_abs_max": 0.0,
        "delta_norm": 0.0,
        "delta_smooth": 0.0,
        "gate_mean": 0.0,
        "gate_std": 0.0,
        "gate_min": 0.0,
        "gate_max": 0.0,
        "loss_forecast": 0.0,
        "loss_lora_anchor": 0.0,
        "loss_cal_anchor": 0.0,
        "loss_cal_base_anchor": 0.0,
        "loss_norm": 0.0,
        "loss_smooth": 0.0,
        "loss_gate": 0.0,
        "loss_total": 0.0,
    }
    for idx in range(len(block_ranges)):
        sums[f"gate_mean_b{idx}"] = 0.0
        sums[f"gate_std_b{idx}"] = 0.0

    block_base = [0.0 for _ in block_ranges]
    block_adapted = [0.0 for _ in block_ranges]
    block_gain = [0.0 for _ in block_ranges]
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            z_x = autoencoder.encode(x)
            z_base = base_model(z_x.detach())
            y_base = autoencoder.decode(z_base)

            z_lora = adapted_model(z_x.detach())
            z_cal, delta, diag = calibrator(z_lora, z_x.detach(), z_base.detach())
            y_hat = autoencoder.decode(z_cal)
            losses = compute_losses(args, y_hat, y, z_lora, z_base, z_cal, delta, diag["gate_time"])

            base_obs_mse = F.mse_loss(y_base, y)
            adapted_obs_mse = F.mse_loss(y_hat, y)
            base_mae = F.l1_loss(y_base, y)
            adapted_mae = F.l1_loss(y_hat, y)
            lora_anchor = F.mse_loss(z_lora, z_base.detach())
            cal_to_lora_anchor = F.mse_loss(z_cal, z_lora.detach()) if args.calib_type != "none" else z_cal.new_zeros(())
            cal_base_anchor = F.mse_loss(z_cal, z_base.detach()) if args.calib_type != "none" else z_cal.new_zeros(())
            rel_z_lora = (z_lora - z_base).norm() / z_base.norm().clamp_min(1e-9)
            rel_z_cal = (z_cal - z_base).norm() / z_base.norm().clamp_min(1e-9)
            rel_cal_extra = (z_cal - z_lora).norm() / z_lora.norm().clamp_min(1e-9)

            bsz = x.size(0)
            total_count += bsz
            scalar_metrics = {
                "base_obs_mse": base_obs_mse,
                "adapted_obs_mse": adapted_obs_mse,
                "obs_gain": base_obs_mse - adapted_obs_mse,
                "base_mae": base_mae,
                "adapted_mae": adapted_mae,
                "lora_anchor": lora_anchor,
                "cal_to_lora_anchor": cal_to_lora_anchor,
                "cal_base_anchor": cal_base_anchor,
                "relative_z_lora_change": rel_z_lora,
                "relative_z_cal_change": rel_z_cal,
                "relative_cal_extra_change": rel_cal_extra,
                "delta_abs_mean": diag["delta_abs_mean"],
                "delta_abs_max": diag["delta_abs_max"],
                "delta_norm": diag["delta_norm"],
                "delta_smooth": diag["delta_smooth"],
                "gate_mean": diag["gate_mean"],
                "gate_std": diag["gate_std"],
                "gate_min": diag["gate_min"],
                "gate_max": diag["gate_max"],
                "loss_forecast": losses["loss_forecast"],
                "loss_lora_anchor": losses["loss_lora_anchor"],
                "loss_cal_anchor": losses["loss_cal_anchor"],
                "loss_cal_base_anchor": losses["loss_cal_base_anchor"],
                "loss_norm": losses["loss_norm"],
                "loss_smooth": losses["loss_smooth"],
                "loss_gate": losses["loss_gate"],
                "loss_total": losses["loss_total"],
            }
            for idx in range(len(block_ranges)):
                scalar_metrics[f"gate_mean_b{idx}"] = diag[f"gate_mean_b{idx}"]
                scalar_metrics[f"gate_std_b{idx}"] = diag[f"gate_std_b{idx}"]

            for key, value in scalar_metrics.items():
                sums[key] += float(value.item()) * bsz

            base_sq = (y_base - y).pow(2)
            adapted_sq = (y_hat - y).pow(2)
            for idx, (start, end) in enumerate(block_ranges):
                base_block = base_sq[:, start:end, :].mean().item()
                adapted_block = adapted_sq[:, start:end, :].mean().item()
                block_base[idx] += base_block * bsz
                block_adapted[idx] += adapted_block * bsz
                block_gain[idx] += (base_block - adapted_block) * bsz

    metrics = {key: value / max(total_count, 1) for key, value in sums.items()}
    metrics["block_base_mse"] = [value / max(total_count, 1) for value in block_base]
    metrics["block_adapted_mse"] = [value / max(total_count, 1) for value in block_adapted]
    metrics["block_gain"] = [value / max(total_count, 1) for value in block_gain]
    return metrics


def epoch_row(epoch, split, metrics, num_blocks):
    row = {
        "epoch": epoch,
        "split": split,
        "base_obs_mse": metrics["base_obs_mse"],
        "adapted_obs_mse": metrics["adapted_obs_mse"],
        "obs_gain": metrics["obs_gain"],
        "base_mae": metrics["base_mae"],
        "adapted_mae": metrics["adapted_mae"],
        "lora_anchor": metrics["lora_anchor"],
        "cal_to_lora_anchor": metrics["cal_to_lora_anchor"],
        "cal_base_anchor": metrics["cal_base_anchor"],
        "relative_z_lora_change": metrics["relative_z_lora_change"],
        "relative_z_cal_change": metrics["relative_z_cal_change"],
        "relative_cal_extra_change": metrics["relative_cal_extra_change"],
        "delta_abs_mean": metrics["delta_abs_mean"],
        "delta_abs_max": metrics["delta_abs_max"],
        "delta_norm": metrics["delta_norm"],
        "delta_smooth": metrics["delta_smooth"],
        "gate_mean": metrics["gate_mean"],
        "gate_std": metrics["gate_std"],
        "gate_min": metrics["gate_min"],
        "gate_max": metrics["gate_max"],
        "loss_forecast": metrics["loss_forecast"],
        "loss_lora_anchor": metrics["loss_lora_anchor"],
        "loss_cal_anchor": metrics["loss_cal_anchor"],
        "loss_cal_base_anchor": metrics["loss_cal_base_anchor"],
        "loss_norm": metrics["loss_norm"],
        "loss_smooth": metrics["loss_smooth"],
        "loss_gate": metrics["loss_gate"],
        "loss_total": metrics["loss_total"],
    }
    for idx in range(num_blocks):
        row[f"gate_mean_b{idx}"] = metrics[f"gate_mean_b{idx}"]
        row[f"gate_std_b{idx}"] = metrics[f"gate_std_b{idx}"]
        row[f"block_base_mse_b{idx}"] = metrics["block_base_mse"][idx]
        row[f"block_adapted_mse_b{idx}"] = metrics["block_adapted_mse"][idx]
        row[f"block_gain_b{idx}"] = metrics["block_gain"][idx]
    return row


def save_checkpoint(path, args, adapted_model, calibrator, epoch, val_metrics, test_metrics, lora_meta):
    torch.save(
        {
            "args": vars(args),
            "epoch": epoch,
            "adapted_model_state_dict": adapted_model.state_dict(),
            "calibrator_state_dict": calibrator.state_dict(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "lora_meta": lora_meta,
        },
        path,
    )


def print_epoch_line(epoch, train_metrics, val_metrics, test_metrics):
    print(
        f"epoch {epoch:03d} | "
        f"train {train_metrics['base_obs_mse']:.6f}->{train_metrics['adapted_obs_mse']:.6f} (gain {train_metrics['obs_gain']:.6f}) | "
        f"val {val_metrics['base_obs_mse']:.6f}->{val_metrics['adapted_obs_mse']:.6f} (gain {val_metrics['obs_gain']:.6f}) | "
        f"test {test_metrics['base_obs_mse']:.6f}->{test_metrics['adapted_obs_mse']:.6f} (gain {test_metrics['obs_gain']:.6f})"
    )
    print(
        f"           mae val/test {val_metrics['base_mae']:.6f}->{val_metrics['adapted_mae']:.6f} | "
        f"{test_metrics['base_mae']:.6f}->{test_metrics['adapted_mae']:.6f}"
    )
    print(
        f"           lora_anchor {val_metrics['lora_anchor']:.6f} | "
        f"cal_lora/base {val_metrics['cal_to_lora_anchor']:.6f}/{val_metrics['cal_base_anchor']:.6f} | "
        f"rel_z_lora/cal/extra {val_metrics['relative_z_lora_change']:.6f}/"
        f"{val_metrics['relative_z_cal_change']:.6f}/{val_metrics['relative_cal_extra_change']:.6f}"
    )
    print(
        f"           delta mean/max {val_metrics['delta_abs_mean']:.6f}/{val_metrics['delta_abs_max']:.6f} | "
        f"norm/smooth {val_metrics['delta_norm']:.6f}/{val_metrics['delta_smooth']:.6f}"
    )
    print(
        f"           gate mean/std/min/max {val_metrics['gate_mean']:.4f}/{val_metrics['gate_std']:.4f}/"
        f"{val_metrics['gate_min']:.4f}/{val_metrics['gate_max']:.4f}"
    )
    print(
        f"           block gain val {format_block_values(val_metrics['block_gain'])} | "
        f"test {format_block_values(test_metrics['block_gain'])}"
    )


def write_final_artifacts(
    args,
    output_dir,
    train_metrics,
    val_metrics,
    test_metrics,
    best_epoch,
    trainable_params,
    frozen_params,
    block_ranges,
    lora_meta,
):
    summary = {
        "best_epoch": best_epoch,
        "model": args.model,
        "pred_len": args.pred_len,
        "use_lora": int(args.use_lora),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "calib_type": args.calib_type,
        "calib_rank": args.calib_rank,
        "calib_gate": args.calib_gate,
        "dynamic_scale": args.dynamic_scale,
        "alpha": args.alpha,
        "test_base_obs_mse": test_metrics["base_obs_mse"],
        "test_adapted_obs_mse": test_metrics["adapted_obs_mse"],
        "test_obs_gain": test_metrics["obs_gain"],
        "test_base_mae": test_metrics["base_mae"],
        "test_adapted_mae": test_metrics["adapted_mae"],
        "test_lora_anchor": test_metrics["lora_anchor"],
        "test_cal_anchor": test_metrics["cal_to_lora_anchor"],
        "test_cal_base_anchor": test_metrics["cal_base_anchor"],
        "test_relative_z_lora_change": test_metrics["relative_z_lora_change"],
        "test_relative_z_cal_change": test_metrics["relative_z_cal_change"],
        "test_relative_cal_extra_change": test_metrics["relative_cal_extra_change"],
        "gate_mean": test_metrics["gate_mean"],
        "gate_std": test_metrics["gate_std"],
        "gate_min": test_metrics["gate_min"],
        "gate_max": test_metrics["gate_max"],
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "lora_num_target_modules": lora_meta["num_target_modules"],
        "lora_target_modules": lora_meta["target_modules"],
    }
    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    diagnostics = {
        "train": detach_metrics(train_metrics),
        "val": detach_metrics(val_metrics),
        "test": detach_metrics(test_metrics),
        "block_ranges": block_ranges,
        "lora_meta": lora_meta,
    }
    with open(os.path.join(output_dir, "test_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)


def train(args, device):
    set_seed(args.seed)
    sync_args_from_base_checkpoint(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device, freeze=True)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    base_model, _ = load_forecaster_from_checkpoint(
        args,
        device,
        args.base_checkpoint,
        LatentForecaster,
        freeze=True,
        return_checkpoint=True,
    )
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    adapted_model = load_forecaster_from_checkpoint(
        args,
        device,
        args.base_checkpoint,
        LatentForecaster,
        freeze=False,
        return_checkpoint=False,
    )
    lora_meta = inject_lora(
        adapted_model.backbone,
        model_name=args.model,
        use_lora=bool(args.use_lora),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_mode=args.lora_target,
    )
    adapted_model = adapted_model.to(device)
    adapted_model.eval()

    calibrator = ForecastCalibration(
        pred_len=args.pred_len,
        d_model=args.d_model,
        calib_type=args.calib_type,
        calib_rank=args.calib_rank,
        alpha=args.alpha,
        calib_gate=args.calib_gate,
        gate_hidden=args.gate_hidden,
        gate_range=args.gate_range,
        gate_init=args.gate_init,
        num_blocks=args.num_blocks,
    ).to(device)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)
    block_ranges = compute_block_ranges(args.pred_len, args.num_blocks)

    trainable_params = count_trainable_params(adapted_model) + count_trainable_params(calibrator)
    frozen_params = (
        count_frozen_params(autoencoder)
        + count_frozen_params(adapted_model)
        + count_frozen_params(calibrator)
    )
    trainable_tensors = [param for param in list(adapted_model.parameters()) + list(calibrator.parameters()) if param.requires_grad]
    optimizer = (
        torch.optim.AdamW(trainable_tensors, lr=args.lr, weight_decay=args.weight_decay)
        if trainable_tensors
        else None
    )

    print(
        f"Forecast-Aware Forecaster-LoRA + Decoder-Compatible Calibration | "
        f"model={args.model} data={args.data} pred_len={args.pred_len}"
    )
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(
        f"LoRA enabled={bool(args.use_lora)} target={args.lora_target} rank={args.lora_rank} "
        f"alpha={args.lora_alpha} dropout={args.lora_dropout}"
    )
    print(
        f"Calibration type={args.calib_type} calib_rank={args.calib_rank} alpha={args.alpha} "
        f"calib_gate={args.calib_gate}"
    )
    print(f"LoRA target modules ({lora_meta['num_target_modules']}): {lora_meta['target_modules']}")
    print(f"Trainable params: {trainable_params} | Frozen params: {frozen_params}")
    print(
        f"Hyperparams | lambda_lora_anchor={args.lambda_lora_anchor} "
        f"lambda_cal_anchor={args.lambda_cal_anchor} lambda_cal_base_anchor={args.lambda_cal_base_anchor} "
        f"lambda_norm={args.lambda_norm} lambda_smooth={args.lambda_smooth} "
        f"lambda_gate={args.lambda_gate} lr={args.lr}"
    )

    history_rows = []
    csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
    best_val = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pth")

    if optimizer is None:
        train_metrics = evaluate(args, autoencoder, base_model, adapted_model, calibrator, eval_train_loader, device, block_ranges)
        val_metrics = evaluate(args, autoencoder, base_model, adapted_model, calibrator, val_loader, device, block_ranges)
        test_metrics = evaluate(args, autoencoder, base_model, adapted_model, calibrator, test_loader, device, block_ranges)
        history_rows.extend(
            [
                epoch_row(0, "train", train_metrics, len(block_ranges)),
                epoch_row(0, "val", val_metrics, len(block_ranges)),
                epoch_row(0, "test", test_metrics, len(block_ranges)),
            ]
        )
        pd.DataFrame(history_rows).to_csv(csv_path, index=False)
        save_checkpoint(best_checkpoint_path, args, adapted_model, calibrator, 0, val_metrics, test_metrics, lora_meta)
        write_final_artifacts(
            args,
            args.output_dir,
            train_metrics,
            val_metrics,
            test_metrics,
            best_epoch=0,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            block_ranges=block_ranges,
            lora_meta=lora_meta,
        )
        print(
            f"Baseline evaluation [test]\n"
            f"  obs MSE / MAE: {test_metrics['base_obs_mse']:.6f} / {test_metrics['base_mae']:.6f} -> "
            f"{test_metrics['adapted_obs_mse']:.6f} / {test_metrics['adapted_mae']:.6f}\n"
            f"  obs gain: {test_metrics['obs_gain']:.6f}\n"
            f"  lora_anchor {test_metrics['lora_anchor']:.6f} | "
            f"cal_lora/base {test_metrics['cal_to_lora_anchor']:.6f}/{test_metrics['cal_base_anchor']:.6f}\n"
            f"  block gain: {format_block_values(test_metrics['block_gain'])}"
        )
        return

    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        adapted_model.train()
        calibrator.train()

        for batch_x, batch_y, _, _ in train_loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)

            with torch.no_grad():
                z_x = autoencoder.encode(x)
                z_base = base_model(z_x.detach())

            z_lora = adapted_model(z_x.detach())
            z_cal, delta, diag = calibrator(z_lora, z_x.detach(), z_base.detach())
            y_hat = autoencoder.decode(z_cal)
            losses = compute_losses(args, y_hat, y, z_lora, z_base, z_cal, delta, diag["gate_time"])

            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_tensors, args.grad_clip)
            optimizer.step()

        train_metrics = evaluate(args, autoencoder, base_model, adapted_model, calibrator, eval_train_loader, device, block_ranges)
        val_metrics = evaluate(args, autoencoder, base_model, adapted_model, calibrator, val_loader, device, block_ranges)
        test_metrics = evaluate(args, autoencoder, base_model, adapted_model, calibrator, test_loader, device, block_ranges)
        print_epoch_line(epoch, train_metrics, val_metrics, test_metrics)

        history_rows.extend(
            [
                epoch_row(epoch, "train", train_metrics, len(block_ranges)),
                epoch_row(epoch, "val", val_metrics, len(block_ranges)),
                epoch_row(epoch, "test", test_metrics, len(block_ranges)),
            ]
        )
        pd.DataFrame(history_rows).to_csv(csv_path, index=False)

        current_val = val_metrics["adapted_obs_mse"]
        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_checkpoint_path, args, adapted_model, calibrator, epoch, val_metrics, test_metrics, lora_meta)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    adapted_model.load_state_dict(checkpoint["adapted_model_state_dict"])
    calibrator.load_state_dict(checkpoint["calibrator_state_dict"])
    adapted_model.eval()
    calibrator.eval()

    final_train = evaluate(args, autoencoder, base_model, adapted_model, calibrator, eval_train_loader, device, block_ranges)
    final_val = evaluate(args, autoencoder, base_model, adapted_model, calibrator, val_loader, device, block_ranges)
    final_test = evaluate(args, autoencoder, base_model, adapted_model, calibrator, test_loader, device, block_ranges)

    write_final_artifacts(
        args,
        args.output_dir,
        final_train,
        final_val,
        final_test,
        best_epoch=best_epoch,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        block_ranges=block_ranges,
        lora_meta=lora_meta,
    )

    print(
        f"Forecast-aware latent adaptation [test]\n"
        f"  obs MSE / MAE: {final_test['base_obs_mse']:.6f} / {final_test['base_mae']:.6f} -> "
        f"{final_test['adapted_obs_mse']:.6f} / {final_test['adapted_mae']:.6f}\n"
        f"  obs gain: {final_test['obs_gain']:.6f}\n"
        f"  lora_anchor {final_test['lora_anchor']:.6f} | "
        f"cal_lora/base {final_test['cal_to_lora_anchor']:.6f}/{final_test['cal_base_anchor']:.6f}\n"
        f"  rel_z_lora/cal/extra {final_test['relative_z_lora_change']:.6f}/"
        f"{final_test['relative_z_cal_change']:.6f}/{final_test['relative_cal_extra_change']:.6f}\n"
        f"  gate mean/std/min/max {final_test['gate_mean']:.4f}/{final_test['gate_std']:.4f}/"
        f"{final_test['gate_min']:.4f}/{final_test['gate_max']:.4f}\n"
        f"  block gain: {format_block_values(final_test['block_gain'])}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Forecast-aware Forecaster-LoRA with decoder-compatible latent calibration"
    )
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

    parser.add_argument("--model", type=str, required=True, choices=["DLinear", "PatchTST", "iTransformer"])
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_lora", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target",
        type=str,
        default="auto",
        choices=["auto", "all_linear", "head", "attention_only", "qv_only"],
    )

    parser.add_argument(
        "--calib_type",
        type=str,
        default="lowrank",
        choices=["none", "global", "block", "lowrank"],
    )
    parser.add_argument("--calib_rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--num_blocks", type=int, default=4)

    parser.add_argument(
        "--calib_gate",
        type=str,
        default=None,
        choices=["none", "scalar", "block"],
    )
    parser.add_argument(
        "--dynamic_scale",
        type=str,
        default=None,
        choices=["none", "scalar", "block"],
        help="Deprecated alias for calib_gate.",
    )
    parser.add_argument("--gate_hidden", type=int, default=64)
    parser.add_argument("--gate_range", type=float, default=1.0)
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--scale_hidden", type=int, default=None, help="Deprecated alias for gate_hidden.")
    parser.add_argument("--scale_range", type=float, default=None, help="Deprecated alias for gate_range.")

    parser.add_argument("--lambda_lora_anchor", type=float, default=0.1)
    parser.add_argument("--lambda_cal_anchor", type=float, default=0.5)
    parser.add_argument("--lambda_cal_base_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_norm", type=float, default=0.005)
    parser.add_argument("--lambda_smooth", type=float, default=0.01)
    parser.add_argument("--lambda_gate", type=float, default=0.01)
    parser.add_argument("--lambda_scale", type=float, default=None, help="Deprecated alias for lambda_gate.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.calib_gate is None:
        if args.dynamic_scale is not None:
            args.calib_gate = args.dynamic_scale
        else:
            args.calib_gate = "scalar"
    if args.dynamic_scale is None:
        args.dynamic_scale = args.calib_gate
    if args.scale_hidden is not None:
        args.gate_hidden = args.scale_hidden
    if args.scale_range is not None:
        args.gate_range = args.scale_range
    if args.lambda_scale is not None:
        args.lambda_gate = args.lambda_scale
    device = resolve_device(args.device)
    train(args, device)


if __name__ == "__main__":
    main()
