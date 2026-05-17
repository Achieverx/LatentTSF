import argparse
import json
import os
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
import torch
import torch.nn.functional as F
from datasets import load_dataset  # noqa: F401
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from train_frozen_dlinear_branch_selector import FrozenDLinearMultiBranch, LatentForecaster


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return model, checkpoint["model_state_dict"]


def extract_split(args, autoencoder, baseline, device, flag):
    loader = official_loader(args, flag)
    z_x_all, z_y_all, y_all, z_base_all, y_base_all = [], [], [], [], []
    print(f"\nExtract {flag} latents and baseline residuals...", flush=True)
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len :, :].float().to(device)
            z_x = autoencoder.encode(x)
            z_y = autoencoder.encode(y)
            z_base = baseline(z_x)
            y_base = autoencoder.decode(z_base)
            z_x_all.append(z_x.cpu().numpy().astype(np.float32))
            z_y_all.append(z_y.cpu().numpy().astype(np.float32))
            y_all.append(y.cpu().numpy().astype(np.float32))
            z_base_all.append(z_base.cpu().numpy().astype(np.float32))
            y_base_all.append(y_base.cpu().numpy().astype(np.float32))

    split = {
        "z_x": np.concatenate(z_x_all, axis=0),
        "z_y": np.concatenate(z_y_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
        "z_base": np.concatenate(z_base_all, axis=0),
        "y_base": np.concatenate(y_base_all, axis=0),
    }
    print(
        f"  {flag}: z_x {split['z_x'].shape}, z_y {split['z_y'].shape}, "
        f"baseline MSE {np.mean((split['y_base'] - split['y']) ** 2):.6f}",
        flush=True,
    )
    return split


def standardize_block(values, mode):
    flat = values.reshape(values.shape[0], -1).astype(np.float32)
    if mode == "sample_standard":
        flat = flat - flat.mean(axis=1, keepdims=True)
        flat = flat / (flat.std(axis=1, keepdims=True) + 1e-6)
    elif mode == "global_standard":
        flat = flat - flat.mean(axis=0, keepdims=True)
        flat = flat / (flat.std(axis=0, keepdims=True) + 1e-6)
    elif mode != "none":
        raise ValueError(f"Unknown cluster_normalize={mode}")
    return flat


def flatten_for_cluster(args, split):
    if args.cluster_target == "delta_z":
        values = split["z_y"] - split["z_base"]
    elif args.cluster_target == "obs_residual":
        values = split["y"] - split["y_base"]
    elif args.cluster_target in ["joint_delta_z", "joint_obs_residual"]:
        if args.cluster_target == "joint_delta_z":
            residual = split["z_y"] - split["z_base"]
        else:
            residual = split["y"] - split["y_base"]
        residual_flat = standardize_block(residual, args.cluster_normalize)
        history_flat = standardize_block(split["z_x"], args.cluster_normalize)

        # Normalize each block by dimensionality so lambda controls the distance tradeoff,
        # instead of whichever block has more coordinates dominating KMeans.
        residual_flat = residual_flat / np.sqrt(max(residual_flat.shape[1], 1))
        history_flat = history_flat / np.sqrt(max(history_flat.shape[1], 1))
        residual_weight = np.sqrt(args.predictability_lambda)
        history_weight = np.sqrt(1.0 - args.predictability_lambda)
        return np.concatenate(
            [residual_weight * residual_flat, history_weight * history_flat],
            axis=1,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown cluster_target={args.cluster_target}")
    return standardize_block(values, args.cluster_normalize)


def fit_residual_kmeans(args, splits, out_dir):
    train_flat = flatten_for_cluster(args, splits["train"])
    kmeans = KMeans(
        n_clusters=args.num_branches,
        init="k-means++",
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=args.seed,
    )
    labels = {"train": kmeans.fit_predict(train_flat).astype(np.int64)}
    joblib.dump(kmeans, os.path.join(out_dir, f"kmeans_{args.cluster_target}_K{args.num_branches}.joblib"))

    for flag in ["val", "test"]:
        labels[flag] = kmeans.predict(flatten_for_cluster(args, splits[flag])).astype(np.int64)

    summary = {}
    for flag, lab in labels.items():
        counts = np.bincount(lab, minlength=args.num_branches)
        summary[flag] = {"counts": counts.tolist(), "ratios": (counts / max(len(lab), 1)).tolist()}
        np.save(os.path.join(out_dir, f"{flag}_labels_K{args.num_branches}.npy"), lab)

    with open(os.path.join(out_dir, "residual_kmeans_label_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResidual-pattern KMeans labels:")
    print(json.dumps(summary, indent=2), flush=True)
    return labels, summary


def make_loader(split, labels, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split["z_x"]),
            torch.from_numpy(split["z_y"]),
            torch.from_numpy(split["y"]),
            torch.from_numpy(split["y_base"]),
            torch.from_numpy(labels),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def decode_branches(autoencoder, z_branches):
    bsz, branches, steps, dim = z_branches.shape
    y_branches = autoencoder.decode(z_branches.reshape(bsz * branches, steps, dim))
    return y_branches.reshape(bsz, branches, steps, -1)


def branch_errors(branches, target):
    return ((branches - target[:, None]) ** 2).mean(dim=(2, 3))


def pairwise_diversity(y_branches):
    if y_branches.size(1) < 2:
        return y_branches.new_zeros(())
    flat = y_branches.flatten(start_dim=2)
    values = []
    for i in range(y_branches.size(1)):
        for j in range(i + 1, y_branches.size(1)):
            values.append(((flat[:, i] - flat[:, j]) ** 2).mean(dim=1))
    return torch.stack(values, dim=1).mean()


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def trainable_parameters(model, freeze_backbone):
    for name, param in model.named_parameters():
        param.requires_grad = (not freeze_backbone) or (not name.startswith("backbone."))
    return [param for param in model.parameters() if param.requires_grad]


def evaluate(args, model, autoencoder, loader, device):
    model.eval()
    sums = {
        "loss": 0.0,
        "assign_latent_mse": 0.0,
        "best_latent_mse": 0.0,
        "score_loss": 0.0,
        "baseline_obs_mse": 0.0,
        "oracle_obs_mse": 0.0,
        "oracle_obs_mae": 0.0,
        "selected_obs_mse": 0.0,
        "fused_obs_mse": 0.0,
        "branch_diversity": 0.0,
        "winner_at_1": 0.0,
        "winner_at_3": 0.0,
    }
    total = 0
    winner_counts = np.zeros(args.num_branches, dtype=np.int64)
    with torch.no_grad():
        for z_x, z_y, y, y_base_ref, labels in loader:
            z_x = z_x.to(device)
            z_y = z_y.to(device)
            y = y.to(device)
            y_base_ref = y_base_ref.to(device)
            labels = labels.to(device)
            z_base, z_branches, logits = model(z_x)
            batch_idx = torch.arange(z_x.size(0), device=device)
            assign = F.mse_loss(z_branches[batch_idx, labels], z_y)
            latent_errors = branch_errors(z_branches, z_y)
            best_latent = latent_errors.min(dim=-1).values.mean()
            q = torch.softmax(-latent_errors.detach() / args.score_tau, dim=-1)
            score_loss = soft_cross_entropy(logits, q)
            loss = args.assign_weight * assign + args.best_weight * best_latent + args.score_weight * score_loss

            y_branches = decode_branches(autoencoder, z_branches)
            probs = torch.softmax(logits, dim=-1)
            selected = logits.argmax(dim=-1)
            y_selected = y_branches[batch_idx, selected]
            y_fused = torch.einsum("bk,bktc->btc", probs, y_branches)
            obs_errors = ((y_branches - y[:, None]) ** 2).mean(dim=(2, 3))
            obs_mae = (y_branches - y[:, None]).abs().mean(dim=(2, 3))
            winners = obs_errors.argmin(dim=-1)
            top3 = logits.topk(min(3, args.num_branches), dim=-1).indices

            bsz = z_x.size(0)
            total += bsz
            sums["loss"] += loss.item() * bsz
            sums["assign_latent_mse"] += assign.item() * bsz
            sums["best_latent_mse"] += best_latent.item() * bsz
            sums["score_loss"] += score_loss.item() * bsz
            sums["baseline_obs_mse"] += ((y_base_ref - y) ** 2).mean(dim=(1, 2)).sum().item()
            sums["oracle_obs_mse"] += obs_errors[batch_idx, winners].sum().item()
            sums["oracle_obs_mae"] += obs_mae[batch_idx, winners].sum().item()
            sums["selected_obs_mse"] += ((y_selected - y) ** 2).mean(dim=(1, 2)).sum().item()
            sums["fused_obs_mse"] += ((y_fused - y) ** 2).mean(dim=(1, 2)).sum().item()
            sums["branch_diversity"] += pairwise_diversity(y_branches).item() * bsz
            sums["winner_at_1"] += (selected == winners).float().sum().item()
            sums["winner_at_3"] += (top3 == winners[:, None]).any(dim=-1).float().sum().item()
            winner_counts += np.bincount(winners.cpu().numpy(), minlength=args.num_branches)

    result = {key: value / max(total, 1) for key, value in sums.items()}
    result["winner_counts"] = winner_counts.tolist()
    return result


def train(args, splits, labels, autoencoder, latent_state, device, out_dir):
    train_loader = make_loader(splits["train"], labels["train"], args.batch_size, True)
    val_loader = make_loader(splits["val"], labels["val"], args.batch_size, False)
    test_loader = make_loader(splits["test"], labels["test"], args.batch_size, False)

    model = FrozenDLinearMultiBranch(args).to(device)
    missing, unexpected = model.load_state_dict(latent_state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("proto")]
    if unexpected:
        raise RuntimeError(f"Unexpected keys while initializing backbone: {unexpected}")
    optimizer = torch.optim.AdamW(trainable_parameters(model, args.freeze_backbone), lr=args.lr, weight_decay=args.weight_decay)
    best_path = os.path.join(out_dir, "best_multibranch_official.pt")
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"\nTrain residual-pattern branch bank | target={args.cluster_target} "
        f"normalize={args.cluster_normalize} predictability_lambda={args.predictability_lambda} "
        f"freeze_backbone={args.freeze_backbone}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for z_x, z_y, y, y_base_ref, labels in train_loader:
            z_x = z_x.to(device)
            z_y = z_y.to(device)
            labels = labels.to(device)
            z_base, z_branches, logits = model(z_x)
            batch_idx = torch.arange(z_x.size(0), device=device)
            assign = F.mse_loss(z_branches[batch_idx, labels], z_y)
            latent_errors = branch_errors(z_branches, z_y)
            best_latent = latent_errors.min(dim=-1).values.mean()
            q = torch.softmax(-latent_errors.detach() / args.score_tau, dim=-1)
            score_loss = soft_cross_entropy(logits, q)
            loss = args.assign_weight * assign + args.best_weight * best_latent + args.score_weight * score_loss
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model, args.freeze_backbone), args.grad_clip)
            optimizer.step()

        train_metrics = evaluate(args, model, autoencoder, train_loader, device)
        val_metrics = evaluate(args, model, autoencoder, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | train oracle {train_metrics['oracle_obs_mse']:.6f} "
            f"base {train_metrics['baseline_obs_mse']:.6f} | val oracle {val_metrics['oracle_obs_mse']:.6f} "
            f"base {val_metrics['baseline_obs_mse']:.6f} selected {val_metrics['selected_obs_mse']:.6f} "
            f"w@1 {val_metrics['winner_at_1']:.3f}",
            flush=True,
        )
        metric = val_metrics[args.early_stop_metric]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_metric": best_metric,
                    "early_stop_metric": args.early_stop_metric,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = evaluate(args, model, autoencoder, train_loader, device)
    val_metrics = evaluate(args, model, autoencoder, val_loader, device)
    test_metrics = evaluate(args, model, autoencoder, test_loader, device)
    summary = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_model_path": best_path,
    }
    with open(os.path.join(out_dir, "residual_pattern_branch_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResidual-pattern branch bank [test]")
    print(f"  baseline obs MSE:  {test_metrics['baseline_obs_mse']:.6f}")
    print(f"  oracle obs MSE:    {test_metrics['oracle_obs_mse']:.6f}")
    print(f"  selected obs MSE:  {test_metrics['selected_obs_mse']:.6f}")
    print(f"  fused obs MSE:     {test_metrics['fused_obs_mse']:.6f}")
    print(f"  diversity obs MSE: {test_metrics['branch_diversity']:.6f}")
    print(f"  winner@1/@3:       {test_metrics['winner_at_1']:.4f} / {test_metrics['winner_at_3']:.4f}")
    print(f"  winner counts:     {test_metrics['winner_counts']}")


def main():
    parser = argparse.ArgumentParser(description="Build a branch bank by clustering residual improvement patterns")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/residual_pattern_multibranch_ETTh1_sl96_pl96_deltaZ_K8")
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
    parser.add_argument("--num_branches", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--cluster_target",
        type=str,
        default="delta_z",
        choices=["delta_z", "obs_residual", "joint_delta_z", "joint_obs_residual"],
    )
    parser.add_argument("--cluster_normalize", type=str, default="sample_standard", choices=["sample_standard", "global_standard", "none"])
    parser.add_argument("--predictability_lambda", type=float, default=0.5)
    parser.add_argument("--kmeans_n_init", type=int, default=20)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--assign_weight", type=float, default=1.0)
    parser.add_argument("--best_weight", type=float, default=0.1)
    parser.add_argument("--score_weight", type=float, default=0.1)
    parser.add_argument("--score_tau", type=float, default=0.05)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="oracle_obs_mse",
        choices=["loss", "assign_latent_mse", "best_latent_mse", "oracle_obs_mse", "selected_obs_mse"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    autoencoder = load_autoencoder(args, device)
    baseline, latent_state = load_latenttsf(args, device)
    splits = {flag: extract_split(args, autoencoder, baseline, device, flag) for flag in ["train", "val", "test"]}
    for flag, split in splits.items():
        for key, value in split.items():
            np.save(os.path.join(args.output_dir, f"{flag}_{key}.npy"), value)
    labels, label_summary = fit_residual_kmeans(args, splits, args.output_dir)
    train(args, splits, labels, autoencoder, latent_state, device, args.output_dir)


if __name__ == "__main__":
    main()
