import argparse
import json
import os
from copy import deepcopy

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

CHANNEL_MODE_CHOICES = [
    "original",
    "zero_mix",
    "real_corr",
    "random_corr",
    "identity_corr",
    "no_anchor",
    "no_corr_loss",
]


def batch_corr(x, eps):
    centered = x - x.mean(dim=1, keepdim=True)
    denom = torch.sqrt(centered.pow(2).mean(dim=1, keepdim=True) + eps)
    normalized = centered / denom
    corr = torch.einsum("blc,bld->bcd", normalized, normalized) / max(x.size(1), 1)
    eye = torch.eye(x.size(-1), device=x.device, dtype=x.dtype).unsqueeze(0)
    return corr * (1.0 - eye)


class ChannelMix(nn.Module):
    def __init__(self, channels, gamma_init, channel_mode, corr_eps=1e-6, use_channel_projection=False):
        super().__init__()
        self.channels = channels
        self.channel_mode = channel_mode
        self.corr_eps = float(corr_eps)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.use_channel_projection = bool(use_channel_projection)
        if self.use_channel_projection:
            self.channel_projection = nn.Linear(channels, channels, bias=False)
            nn.init.eye_(self.channel_projection.weight)
        else:
            self.channel_projection = None

    def _resolve_corr(self, x):
        base_corr = batch_corr(x, self.corr_eps)

        if self.channel_mode in {"original", "zero_mix"}:
            return None, base_corr
        if self.channel_mode == "identity_corr":
            eye = torch.eye(self.channels, device=x.device, dtype=x.dtype).unsqueeze(0)
            return eye, base_corr
        if self.channel_mode == "random_corr":
            if x.size(0) > 1:
                if self.training:
                    perm = torch.randperm(x.size(0), device=x.device)
                else:
                    perm = torch.roll(torch.arange(x.size(0), device=x.device), shifts=1)
                return base_corr[perm], base_corr

            channel_perm = torch.randperm(self.channels, device=x.device)
            return base_corr[:, channel_perm][:, :, channel_perm], base_corr

        return base_corr, base_corr

    def forward(self, x):
        mix_corr, real_corr = self._resolve_corr(x)
        if mix_corr is None:
            x_graph = torch.zeros_like(x)
            x_mixed = x
        else:
            x_graph = torch.einsum("blc,bcd->bld", x, mix_corr)
            if self.channel_projection is not None:
                x_graph = self.channel_projection(x_graph)
            x_mixed = x + self.gamma.to(dtype=x.dtype) * x_graph

        input_delta = x_mixed - x
        stats = {
            "gamma": self.gamma.detach(),
            "relative_input_change": input_delta.abs().mean() / x.abs().mean().clamp_min(self.corr_eps),
            "input_change_abs_mean": input_delta.abs().mean(),
            "corr_abs_mean": real_corr.abs().mean(),
        }
        return x_mixed, real_corr, stats


def load_latenttsf(args, device):
    return load_forecaster_from_checkpoint(
        args,
        device,
        args.latenttsf_checkpoint,
        LatentForecaster,
        return_checkpoint=True,
    )


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


def set_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad = flag


def find_last_trainable_leaf(module):
    preferred = []
    fallback = []
    for _, submodule in module.named_modules():
        params = list(submodule.parameters(recurse=False))
        if not params:
            continue
        fallback.append(submodule)
        if isinstance(submodule, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            preferred.append(submodule)
    if preferred:
        return preferred[-1]
    if fallback:
        return fallback[-1]
    return None


def get_submodule(model, attr_name):
    return getattr(model.module, attr_name) if hasattr(model, "module") else getattr(model, attr_name)


def configure_trainable_modules(args, tuned_autoencoder, tuned_forecaster, channel_mix):
    trainable = []
    module_report = {}

    set_requires_grad(tuned_autoencoder, False)
    set_requires_grad(tuned_forecaster, False)
    set_requires_grad(channel_mix, True)
    trainable.extend(list(channel_mix.parameters()))
    module_report["channel_mix"] = ["gamma"] + (
        ["channel_projection.weight"] if channel_mix.channel_projection is not None else []
    )

    encoder_module = None
    if args.train_encoder_last:
        encoder = get_submodule(tuned_autoencoder, "encoder")
        encoder_module = find_last_trainable_leaf(encoder)
        if encoder_module is not None:
            set_requires_grad(encoder_module, True)
            trainable.extend(list(encoder_module.parameters()))
            module_report["ae_encoder_last"] = [name for name, _ in encoder_module.named_parameters(recurse=False)]

    forecaster_module = None
    if args.train_forecaster_last:
        backbone = get_submodule(tuned_forecaster, "backbone")
        forecaster_module = find_last_trainable_leaf(backbone)
        if forecaster_module is not None:
            set_requires_grad(forecaster_module, True)
            trainable.extend(list(forecaster_module.parameters()))
            module_report["forecaster_last"] = [name for name, _ in forecaster_module.named_parameters(recurse=False)]

    unique_trainable = []
    seen = set()
    for param in trainable:
        if id(param) in seen:
            continue
        seen.add(id(param))
        unique_trainable.append(param)
    return unique_trainable, module_report, encoder_module, forecaster_module


def frozen_eval(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def forward_channel_posttrain_batch(
    args,
    ref_autoencoder,
    ref_forecaster,
    tuned_autoencoder,
    tuned_forecaster,
    channel_mix,
    batch_x,
    batch_y,
):
    x = batch_x.float()
    y = batch_y.float()
    y_true_full = y[:, -args.pred_len :, :]

    with torch.no_grad():
        z_old = ref_autoencoder.encode(x)
        z_base = ref_autoencoder.encode(x)
        z_pred_base = ref_forecaster(z_base)
        y_base_full = ref_autoencoder.decode(z_pred_base)

    x_mixed, corr_x, mix_stats = channel_mix(x)
    z_new = tuned_autoencoder.encode(x_mixed)
    z_pred = tuned_forecaster(z_new)
    y_hat_full = tuned_autoencoder.decode(z_pred)
    x_recon = tuned_autoencoder.decode(z_new)

    y_base_full = y_base_full[:, -args.pred_len :, :]
    y_hat_full = y_hat_full[:, -args.pred_len :, :]

    f_dim = -1 if args.features == "MS" else 0
    y_base = y_base_full[:, :, f_dim:]
    y_hat = y_hat_full[:, :, f_dim:]
    y_true = y_true_full[:, :, f_dim:]
    corr_y_hat = batch_corr(y_hat_full, args.corr_eps)
    corr_y_true = batch_corr(y_true_full, args.corr_eps)

    relative_z_change = (z_new - z_old).abs().mean() / z_old.abs().mean().clamp_min(args.corr_eps)
    stats = {
        "gamma": mix_stats["gamma"],
        "relative_input_change": mix_stats["relative_input_change"],
        "input_change_abs_mean": mix_stats["input_change_abs_mean"],
        "relative_z_change": relative_z_change,
        "corr_abs_mean": mix_stats["corr_abs_mean"],
    }

    return {
        "x": x,
        "x_mixed": x_mixed,
        "y_true": y_true,
        "y_true_full": y_true_full,
        "y_base": y_base,
        "y_hat": y_hat,
        "y_base_full": y_base_full,
        "y_hat_full": y_hat_full,
        "z_old": z_old,
        "z_new": z_new,
        "z_pred": z_pred,
        "x_recon": x_recon,
        "corr_x": corr_x,
        "corr_y_hat": corr_y_hat,
        "corr_y_true": corr_y_true,
        "stats": stats,
    }


def evaluate_loader(
    args,
    ref_autoencoder,
    ref_forecaster,
    tuned_autoencoder,
    tuned_forecaster,
    channel_mix,
    dataset,
    loader,
    device,
):
    channel_mix.eval()
    tuned_autoencoder.eval()
    tuned_forecaster.eval()

    preds_base = []
    preds_post = []
    trues = []
    sums = {
        "base_loss": 0.0,
        "pred_loss": 0.0,
        "forecast_loss": 0.0,
        "rec_loss": 0.0,
        "anchor_loss": 0.0,
        "corr_loss": 0.0,
        "gamma": 0.0,
        "relative_input_change": 0.0,
        "relative_z_change": 0.0,
        "input_change_abs_mean": 0.0,
        "corr_abs_mean": 0.0,
    }
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().to(device)
            y = batch_y.float().to(device)
            batch = forward_channel_posttrain_batch(
                args,
                ref_autoencoder,
                ref_forecaster,
                tuned_autoencoder,
                tuned_forecaster,
                channel_mix,
                x,
                y,
            )

            base_loss = F.mse_loss(batch["y_base"], batch["y_true"])
            pred_loss = F.mse_loss(batch["y_hat"], batch["y_true"])
            forecast_loss = pred_loss
            rec_loss = F.mse_loss(batch["x_recon"], batch["x"])
            anchor_loss = F.mse_loss(batch["z_new"], batch["z_old"])
            corr_loss = F.mse_loss(batch["corr_y_hat"], batch["corr_y_true"])

            bsz = x.size(0)
            total_count += bsz
            sums["base_loss"] += base_loss.item() * bsz
            sums["pred_loss"] += pred_loss.item() * bsz
            sums["forecast_loss"] += forecast_loss.item() * bsz
            sums["rec_loss"] += rec_loss.item() * bsz
            sums["anchor_loss"] += anchor_loss.item() * bsz
            sums["corr_loss"] += corr_loss.item() * bsz
            for key in ["gamma", "relative_input_change", "relative_z_change", "input_change_abs_mean", "corr_abs_mean"]:
                sums[key] += batch["stats"][key].item() * bsz

            y_base_np = batch["y_base"].detach().cpu().numpy()
            y_hat_np = batch["y_hat"].detach().cpu().numpy()
            y_true_np = batch["y_true"].detach().cpu().numpy()
            if getattr(dataset, "scale", False) and args.inverse:
                shape = y_true_np.shape
                y_base_np = inverse_transform_array(dataset, y_base_np, shape)
                y_hat_np = inverse_transform_array(dataset, y_hat_np, shape)
                y_true_np = dataset.inverse_transform(y_true_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

            preds_base.append(y_base_np)
            preds_post.append(y_hat_np)
            trues.append(y_true_np)

    preds_base = np.concatenate(preds_base, axis=0)
    preds_post = np.concatenate(preds_post, axis=0)
    trues = np.concatenate(trues, axis=0)

    base_mae, base_mse, _, _, _ = metric(preds_base, trues)
    post_mae, post_mse, _, _, _ = metric(preds_post, trues)
    result = {
        "base_mse": base_mse,
        "base_mae": base_mae,
        "post_mse": post_mse,
        "post_mae": post_mae,
        "gain_mse": base_mse - post_mse,
        "gain_mae": base_mae - post_mae,
    }
    for key, value in sums.items():
        result[key] = value / max(total_count, 1)
    return result


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    ref_forecaster, checkpoint = load_latenttsf(args, device)
    checkpoint_args, overwritten = prepare_args(args, checkpoint)
    expected_metrics = load_expected_metrics(args, args.latenttsf_checkpoint)

    ref_autoencoder = load_autoencoder(args, device, freeze=True)
    tuned_autoencoder = load_autoencoder(args, device, freeze=True)
    tuned_forecaster = load_forecaster_from_checkpoint(
        args,
        device,
        args.latenttsf_checkpoint,
        LatentForecaster,
        freeze=True,
    )

    frozen_eval(ref_autoencoder)
    frozen_eval(ref_forecaster)
    frozen_eval(tuned_autoencoder)
    frozen_eval(tuned_forecaster)

    channel_mix = ChannelMix(
        channels=args.enc_in,
        gamma_init=args.gamma_init,
        channel_mode=args.channel_mode,
        corr_eps=args.corr_eps,
        use_channel_projection=args.use_channel_projection,
    ).to(device)

    trainable_params, module_report, encoder_module, forecaster_module = configure_trainable_modules(
        args,
        tuned_autoencoder,
        tuned_forecaster,
        channel_mix,
    )
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    eval_train_loader = official_loader(args, "train", shuffle=False)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)
    train_data = eval_train_loader.dataset
    val_data = val_loader.dataset
    test_data = test_loader.dataset

    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "checkpoint_args": checkpoint_args,
                "checkpoint_overrides": overwritten,
                "expected_metrics": expected_metrics,
                "trainable_modules": module_report,
                "encoder_module_type": None if encoder_module is None else encoder_module.__class__.__name__,
                "forecaster_module_type": None if forecaster_module is None else forecaster_module.__class__.__name__,
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

    anchor_weight = 0.0 if args.channel_mode == "no_anchor" else args.lambda_anchor
    corr_weight = 0.0 if args.channel_mode == "no_corr_loss" else args.lambda_corr

    print(
        f"Channel-aware Post-training | mode={args.channel_mode} model={args.model} "
        f"lr={args.lr} gamma_init={args.gamma_init} lambda_rec={args.lambda_rec} "
        f"lambda_anchor={anchor_weight} lambda_corr={corr_weight}",
        flush=True,
    )
    if overwritten:
        print("Checkpoint-aligned args:", flush=True)
        for key, values in overwritten.items():
            print(f"  {key}: {values['old']} -> {values['new']}", flush=True)
    print("Trainable modules:", flush=True)
    for key, value in module_report.items():
        print(f"  {key}: {', '.join(value) if value else 'none'}", flush=True)

    for epoch in range(1, args.epochs + 1):
        channel_mix.train()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            x = batch_x.float().to(device)
            y = batch_y.float().to(device)
            batch = forward_channel_posttrain_batch(
                args,
                ref_autoencoder,
                ref_forecaster,
                tuned_autoencoder,
                tuned_forecaster,
                channel_mix,
                x,
                y,
            )

            forecast_loss = F.mse_loss(batch["y_hat"], batch["y_true"])
            rec_loss = F.mse_loss(batch["x_recon"], batch["x"])
            anchor_loss = F.mse_loss(batch["z_new"], batch["z_old"])
            corr_loss = F.mse_loss(batch["corr_y_hat"], batch["corr_y_true"])
            loss = (
                forecast_loss
                + args.lambda_rec * rec_loss
                + anchor_weight * anchor_loss
                + corr_weight * corr_loss
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

        train_metrics = evaluate_loader(
            args,
            ref_autoencoder,
            ref_forecaster,
            tuned_autoencoder,
            tuned_forecaster,
            channel_mix,
            train_data,
            eval_train_loader,
            device,
        )
        val_metrics = evaluate_loader(
            args,
            ref_autoencoder,
            ref_forecaster,
            tuned_autoencoder,
            tuned_forecaster,
            channel_mix,
            val_data,
            val_loader,
            device,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch {epoch:03d} | "
            f"train baseline MSE {train_metrics['base_mse']:.6f} | "
            f"train channel-aware MSE {train_metrics['post_mse']:.6f} | "
            f"train gain {train_metrics['gain_mse']:.6f} | "
            f"val baseline MSE {val_metrics['base_mse']:.6f} | "
            f"val channel-aware MSE {val_metrics['post_mse']:.6f} | "
            f"val gain {val_metrics['gain_mse']:.6f}",
            flush=True,
        )
        print(
            f"           gamma {val_metrics['gamma']:.6f} | "
            f"forecast {val_metrics['forecast_loss']:.6f} | "
            f"rec {val_metrics['rec_loss']:.6f} | "
            f"anchor {val_metrics['anchor_loss']:.6f} | "
            f"corr {val_metrics['corr_loss']:.6f} | "
            f"rel_x {val_metrics['relative_input_change']:.6f} | "
            f"rel_z {val_metrics['relative_z_change']:.6f}",
            flush=True,
        )

        if val_metrics["post_mse"] < best_val_mse:
            best_val_mse = val_metrics["post_mse"]
            best_val_base_mse = val_metrics["base_mse"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "channel_mix": deepcopy(channel_mix.state_dict()),
                "tuned_autoencoder": deepcopy(tuned_autoencoder.state_dict()),
                "tuned_forecaster": deepcopy(tuned_forecaster.state_dict()),
            }
            torch.save(
                {
                    "channel_mix_state_dict": best_state["channel_mix"],
                    "autoencoder_state_dict": best_state["tuned_autoencoder"],
                    "forecaster_state_dict": best_state["tuned_forecaster"],
                    "args": vars(args),
                    "epoch": epoch,
                    "val_post_mse": best_val_mse,
                    "val_base_mse": best_val_base_mse,
                },
                os.path.join(args.output_dir, "best_channel_aware_posttrain.pt"),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        channel_mix.load_state_dict(best_state["channel_mix"])
        tuned_autoencoder.load_state_dict(best_state["tuned_autoencoder"])
        tuned_forecaster.load_state_dict(best_state["tuned_forecaster"])

    train_metrics = evaluate_loader(
        args,
        ref_autoencoder,
        ref_forecaster,
        tuned_autoencoder,
        tuned_forecaster,
        channel_mix,
        train_data,
        eval_train_loader,
        device,
    )
    val_metrics = evaluate_loader(
        args,
        ref_autoencoder,
        ref_forecaster,
        tuned_autoencoder,
        tuned_forecaster,
        channel_mix,
        val_data,
        val_loader,
        device,
    )
    test_metrics = evaluate_loader(
        args,
        ref_autoencoder,
        ref_forecaster,
        tuned_autoencoder,
        tuned_forecaster,
        channel_mix,
        test_data,
        test_loader,
        device,
    )

    baseline_valid = True
    baseline_warning = None
    if expected_metrics is not None:
        mse_gap = abs(test_metrics["base_mse"] - expected_metrics["mse"])
        if mse_gap > args.baseline_tol:
            baseline_valid = False
            baseline_warning = (
                f"Baseline mismatch: reproduced test MSE={test_metrics['base_mse']:.8f} "
                f"but expected {expected_metrics['mse']:.8f} from {expected_metrics['source']} "
                f"(gap={mse_gap:.8f}). Channel-aware gain is invalid."
            )
            print(f"WARNING: {baseline_warning}", flush=True)
        elif expected_metrics["mae"] is not None:
            mae_gap = abs(test_metrics["base_mae"] - expected_metrics["mae"])
            if mae_gap > args.baseline_tol:
                baseline_valid = False
                baseline_warning = (
                    f"Baseline mismatch: reproduced test MAE={test_metrics['base_mae']:.8f} "
                    f"but expected {expected_metrics['mae']:.8f} from {expected_metrics['source']} "
                    f"(gap={mae_gap:.8f}). Channel-aware gain is invalid."
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
        "best_val_post_mse": best_val_mse,
        "best_val_base_mse": best_val_base_mse,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "baseline_valid": baseline_valid,
        "baseline_warning": baseline_warning,
        "expected_metrics": expected_metrics,
        "trainable_modules": module_report,
    }
    with open(os.path.join(args.output_dir, "channel_aware_posttrain_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o))

    print("\nChannel-aware Post-training [test]", flush=True)
    print(
        f"  baseline MSE / MAE: {test_metrics['base_mse']:.6f} / {test_metrics['base_mae']:.6f}",
        flush=True,
    )
    print(
        f"  channel-aware MSE / MAE: {test_metrics['post_mse']:.6f} / {test_metrics['post_mae']:.6f}",
        flush=True,
    )
    print(
        f"  gain MSE / MAE: {test_metrics['gain_mse']:.6f} / {test_metrics['gain_mae']:.6f}",
        flush=True,
    )
    print(
        f"  gamma: {test_metrics['gamma']:.6f} | "
        f"relative_input_change: {test_metrics['relative_input_change']:.6f} | "
        f"relative_z_change: {test_metrics['relative_z_change']:.6f}",
        flush=True,
    )
    if baseline_warning is None:
        print("  baseline reproduction check: passed", flush=True)
    else:
        print("  baseline reproduction check: failed", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Channel-aware post-training for dataset-specific LatentTSF checkpoints"
    )
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/channel_aware_posttrain")
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

    parser.add_argument("--channel_mode", type=str, default="real_corr", choices=CHANNEL_MODE_CHOICES)
    parser.add_argument("--gamma_init", type=float, default=0.01)
    parser.add_argument("--corr_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_rec", type=float, default=0.1)
    parser.add_argument("--lambda_anchor", type=float, default=0.1)
    parser.add_argument("--lambda_corr", type=float, default=0.01)
    parser.add_argument("--train_encoder_last", type=int, default=1)
    parser.add_argument("--train_forecaster_last", type=int, default=1)
    parser.add_argument("--use_channel_projection", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    args.train_encoder_last = bool(args.train_encoder_last)
    args.train_forecaster_last = bool(args.train_forecaster_last)
    args.use_channel_projection = bool(args.use_channel_projection)
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
