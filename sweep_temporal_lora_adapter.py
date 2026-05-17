import argparse
import itertools
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from train_latent_proto_regularized import LatentForecaster
from train_latenttsf_temporal_lora_adapter import TemporalLoRAResidual


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_list(text, caster):
    return [caster(item.strip()) for item in text.split(",") if item.strip()]


def parse_ranks(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        rt, rc = item.lower().split("x")
        values.append((int(rt), int(rc)))
    return values


def official_loader(args, flag):
    data_set, _ = data_provider(args, flag)
    return DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
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


def cache_path(args, flag):
    return os.path.join(args.cache_dir, f"{args.data}_sl{args.seq_len}_pl{args.pred_len}_{flag}_baseline.pt")


def build_or_load_cache(args, autoencoder, latenttsf, device):
    os.makedirs(args.cache_dir, exist_ok=True)
    splits = {}
    for flag in ["train", "val", "test"]:
        path = cache_path(args, flag)
        if os.path.exists(path) and not args.rebuild_cache:
            splits[flag] = torch.load(path, map_location="cpu", weights_only=False)
            print(f"Loaded cached {flag}: {path}", flush=True)
            continue

        xs, ys, bases = [], [], []
        loader = official_loader(args, flag)
        print(f"Build baseline cache [{flag}]...", flush=True)
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                x = batch_x.float().to(device)
                y = batch_y[:, -args.pred_len :, :].float().to(device)
                z_x = autoencoder.encode(x)
                z_base = latenttsf(z_x)
                y_base = autoencoder.decode(z_base)
                xs.append(x.cpu())
                ys.append(y.cpu())
                bases.append(y_base.cpu())
        split = {
            "x": torch.cat(xs, dim=0).float(),
            "y": torch.cat(ys, dim=0).float(),
            "y_base": torch.cat(bases, dim=0).float(),
        }
        torch.save(split, path)
        splits[flag] = split
        base_mse = ((split["y_base"] - split["y"]) ** 2).mean().item()
        print(f"  saved {flag}: n={split['x'].size(0)} baseline_mse={base_mse:.6f}", flush=True)
    return splits


def make_loader(split, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(split["x"], split["y"], split["y_base"]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def evaluate(model, loader, device):
    model.eval()
    sums = {
        "base_mse": 0.0,
        "base_mae": 0.0,
        "pred_mse": 0.0,
        "pred_mae": 0.0,
        "gain": 0.0,
        "harmful_ratio": 0.0,
        "delta_abs": 0.0,
        "applied_delta_abs": 0.0,
        "reg": 0.0,
    }
    total = 0
    with torch.no_grad():
        for x, y, y_base in loader:
            x = x.to(device)
            y = y.to(device)
            y_base = y_base.to(device)
            y_hat, delta, core = model(x, y_base)
            applied = model.alpha() * delta
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
            sums["applied_delta_abs"] += applied.abs().mean(dim=(1, 2)).sum().item()
            sums["reg"] += applied.pow(2).mean(dim=(1, 2)).sum().item()
    result = {key: value / max(total, 1) for key, value in sums.items()}
    result["alpha"] = model.alpha().item()
    return result


def train_one(args, splits, device, rank_time, rank_channel, alpha_init, lambda_delta, seed):
    set_seed(seed)
    model = TemporalLoRAResidual(
        args.seq_len,
        args.pred_len,
        args.enc_in,
        rank_time,
        rank_channel,
        args.hidden_dim,
        args.dropout,
        alpha_init,
        args.alpha_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(splits["train"], args.batch_size, True, seed)
    val_loader = make_loader(splits["val"], args.batch_size, False, seed)
    test_loader = make_loader(splits["test"], args.batch_size, False, seed)

    best_metric = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, y_base in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_base = y_base.to(device)
            y_hat, delta, core = model(x, y_base)
            applied = model.alpha() * delta
            loss = F.mse_loss(y_hat, y) + lambda_delta * applied.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        metric = val_metrics["pred_mse"]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    return {
        "rank_time": rank_time,
        "rank_channel": rank_channel,
        "alpha_init": alpha_init,
        "lambda_delta": lambda_delta,
        "seed": seed,
        "best_epoch": best_epoch,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }


def summarize(results):
    grouped = {}
    for row in results:
        key = (row["rank_time"], row["rank_channel"], row["alpha_init"], row["lambda_delta"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (rt, rc, alpha, lam), rows in grouped.items():
        gains = np.array([row["test"]["gain"] for row in rows], dtype=np.float64)
        mses = np.array([row["test"]["pred_mse"] for row in rows], dtype=np.float64)
        harmful = np.array([row["test"]["harmful_ratio"] for row in rows], dtype=np.float64)
        final_alpha = np.array([row["test"]["alpha"] for row in rows], dtype=np.float64)
        delta_abs = np.array([row["test"]["applied_delta_abs"] for row in rows], dtype=np.float64)
        summary.append(
            {
                "rank_time": rt,
                "rank_channel": rc,
                "alpha_init": alpha,
                "lambda_delta": lam,
                "runs": len(rows),
                "mean_test_mse": float(mses.mean()),
                "std_test_mse": float(mses.std(ddof=0)),
                "mean_gain": float(gains.mean()),
                "std_gain": float(gains.std(ddof=0)),
                "mean_harmful_ratio": float(harmful.mean()),
                "mean_final_alpha": float(final_alpha.mean()),
                "mean_applied_delta_abs": float(delta_abs.mean()),
            }
        )
    summary.sort(key=lambda item: item["mean_test_mse"])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Formal sweep for frozen LatentTSF Temporal-LoRA residual adapter")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/temporal_lora_sweep")
    parser.add_argument("--cache_dir", type=str, default="./latent_outputs/temporal_lora_sweep/cache")
    parser.add_argument("--rebuild_cache", action="store_true", default=False)
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
    parser.add_argument("--rank_pairs", type=str, default="2x2,4x1,4x2,4x3,6x2,8x2")
    parser.add_argument("--alpha_inits", type=str, default="0.001,0.003,0.01")
    parser.add_argument("--lambda_deltas", type=str, default="0.0001,0.001,0.01")
    parser.add_argument("--seeds", type=str, default="2021,2022,2023")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=0.2)
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

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    autoencoder = load_autoencoder(args, device)
    latenttsf = load_latenttsf(args, device)
    splits = build_or_load_cache(args, autoencoder, latenttsf, device)

    rank_pairs = parse_ranks(args.rank_pairs)
    alpha_inits = parse_list(args.alpha_inits, float)
    lambda_deltas = parse_list(args.lambda_deltas, float)
    seeds = parse_list(args.seeds, int)
    total = len(rank_pairs) * len(alpha_inits) * len(lambda_deltas) * len(seeds)
    results = []
    print(f"Temporal-LoRA formal sweep: {total} runs", flush=True)
    for idx, ((rt, rc), alpha, lam, seed) in enumerate(
        itertools.product(rank_pairs, alpha_inits, lambda_deltas, seeds),
        start=1,
    ):
        print(f"\n[{idx}/{total}] rt={rt} rc={rc} alpha={alpha} lambda={lam} seed={seed}", flush=True)
        row = train_one(args, splits, device, rt, rc, alpha, lam, seed)
        results.append(row)
        print(
            f"  test MSE {row['test']['pred_mse']:.6f} base {row['test']['base_mse']:.6f} "
            f"gain {row['test']['gain']:.6f} harm {row['test']['harmful_ratio']:.3f} "
            f"alpha {row['test']['alpha']:.5f} applied {row['test']['applied_delta_abs']:.6f}",
            flush=True,
        )
        with open(os.path.join(args.output_dir, "runs.json"), "w") as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summarize(results), f, indent=2)

    summary = summarize(results)
    print("\nTop Temporal-LoRA configs by mean test MSE:")
    for item in summary[:10]:
        print(
            f"  rt={item['rank_time']} rc={item['rank_channel']} alpha={item['alpha_init']} "
            f"lambda={item['lambda_delta']} | mse {item['mean_test_mse']:.6f}±{item['std_test_mse']:.6f} "
            f"gain {item['mean_gain']:.6f}±{item['std_gain']:.6f} harm {item['mean_harmful_ratio']:.3f} "
            f"final_alpha {item['mean_final_alpha']:.5f} applied {item['mean_applied_delta_abs']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
