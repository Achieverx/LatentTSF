import argparse
import json
import os

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from train_multibranch_latent import MultiBranchLatentPredictor


def load_autoencoder(args, device):
    autoencoder = get_autoencoder(args).float().to(device)
    state_dict = torch.load(args.autoencoder_path, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    return autoencoder


def load_multibranch(args, device):
    checkpoint = torch.load(args.multibranch_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    hidden_dim = int(train_args.get("hidden_dim", args.hidden_dim))
    dropout = float(train_args.get("dropout", args.dropout))
    num_branches = int(train_args.get("num_branches", args.num_branches))

    model = MultiBranchLatentPredictor(
        seq_len=args.pred_len,
        d_model=args.d_model,
        num_branches=num_branches,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, num_branches, train_args


def pairwise_branch_diversity(y_branches):
    if y_branches.size(1) < 2:
        return torch.tensor(0.0, device=y_branches.device)
    flat = y_branches.flatten(start_dim=2)
    distances = []
    for i in range(y_branches.size(1)):
        for j in range(i + 1, y_branches.size(1)):
            distances.append(((flat[:, i] - flat[:, j]) ** 2).mean(dim=1))
    return torch.stack(distances, dim=1).mean()


def mse_mae(pred, true):
    return ((pred - true) ** 2).mean().item(), (pred - true).abs().mean().item()


def evaluate(args, flag):
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data_set, data_loader = data_provider(args, flag)
    autoencoder = load_autoencoder(args, device)
    multibranch, num_branches, train_args = load_multibranch(args, device)

    y_branches_all = []
    y_selected_all = []
    y_fused_obs_all = []
    y_fused_latent_all = []
    y_true_all = []
    scores_all = []
    probs_all = []
    selected_all = []

    sum_best_mse = 0.0
    sum_best_mae = 0.0
    sum_top3_mse = 0.0
    sum_top3_mae = 0.0
    sum_selected_mse = 0.0
    sum_selected_mae = 0.0
    sum_fused_obs_mse = 0.0
    sum_fused_obs_mae = 0.0
    sum_fused_latent_mse = 0.0
    sum_fused_latent_mae = 0.0
    sum_entropy = 0.0
    sum_max_prob = 0.0
    sum_diversity = 0.0
    total = 0
    winner_counts = np.zeros(num_branches, dtype=np.int64)
    selected_counts = np.zeros(num_branches, dtype=np.int64)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            y_true = batch_y[:, -args.pred_len :, :]

            z_x = autoencoder.encode(batch_x)
            z_branches, scores = multibranch(z_x)
            probs = torch.softmax(scores, dim=1)

            bsz, branches, steps, dim = z_branches.shape
            y_branches = autoencoder.decode(z_branches.reshape(bsz * branches, steps, dim))
            y_branches = y_branches.reshape(bsz, branches, args.pred_len, args.enc_in)

            selected_idx = scores.argmax(dim=1)
            batch_idx = torch.arange(bsz, device=device)
            y_selected = y_branches[batch_idx, selected_idx]

            y_fused_obs = (probs[:, :, None, None] * y_branches).sum(dim=1)
            z_fused = (probs[:, :, None, None] * z_branches).sum(dim=1)
            y_fused_latent = autoencoder.decode(z_fused)

            mse_by_branch = ((y_branches - y_true[:, None]) ** 2).mean(dim=(2, 3))
            mae_by_branch = (y_branches - y_true[:, None]).abs().mean(dim=(2, 3))
            winner_idx = mse_by_branch.argmin(dim=1)
            best_mse = mse_by_branch[batch_idx, winner_idx]
            best_mae = mae_by_branch[batch_idx, winner_idx]

            topk = min(args.top_k, num_branches)
            top_idx = scores.topk(topk, dim=1).indices
            top_mse = torch.gather(mse_by_branch, 1, top_idx)
            top_mae = torch.gather(mae_by_branch, 1, top_idx)
            top_best_pos = top_mse.argmin(dim=1, keepdim=True)
            top3_best_mse = top_mse.gather(1, top_best_pos).squeeze(1)
            top3_best_mae = top_mae.gather(1, top_best_pos).squeeze(1)

            selected_mse, selected_mae = mse_mae(y_selected, y_true)
            fused_obs_mse, fused_obs_mae = mse_mae(y_fused_obs, y_true)
            fused_latent_mse, fused_latent_mae = mse_mae(y_fused_latent, y_true)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
            max_prob = probs.max(dim=1).values
            diversity = pairwise_branch_diversity(y_branches)

            sum_best_mse += best_mse.sum().item()
            sum_best_mae += best_mae.sum().item()
            sum_top3_mse += top3_best_mse.sum().item()
            sum_top3_mae += top3_best_mae.sum().item()
            sum_selected_mse += selected_mse * bsz
            sum_selected_mae += selected_mae * bsz
            sum_fused_obs_mse += fused_obs_mse * bsz
            sum_fused_obs_mae += fused_obs_mae * bsz
            sum_fused_latent_mse += fused_latent_mse * bsz
            sum_fused_latent_mae += fused_latent_mae * bsz
            sum_entropy += entropy.sum().item()
            sum_max_prob += max_prob.sum().item()
            sum_diversity += diversity.item() * bsz
            total += bsz

            winner_counts += np.bincount(winner_idx.cpu().numpy(), minlength=num_branches)
            selected_counts += np.bincount(selected_idx.cpu().numpy(), minlength=num_branches)

            if args.save_predictions:
                y_branches_all.append(y_branches.cpu().numpy().astype(np.float32))
                y_selected_all.append(y_selected.cpu().numpy().astype(np.float32))
                y_fused_obs_all.append(y_fused_obs.cpu().numpy().astype(np.float32))
                y_fused_latent_all.append(y_fused_latent.cpu().numpy().astype(np.float32))
                y_true_all.append(y_true.cpu().numpy().astype(np.float32))
                scores_all.append(scores.cpu().numpy().astype(np.float32))
                probs_all.append(probs.cpu().numpy().astype(np.float32))
                selected_all.append(selected_idx.cpu().numpy().astype(np.int64))

    metrics = {
        "flag": flag,
        "num_samples": total,
        "num_branches": num_branches,
        "top_k": min(args.top_k, num_branches),
        "multibranch_path": args.multibranch_path,
        "autoencoder_path": args.autoencoder_path,
        "multibranch_train_args": train_args,
        "best_of_8_obs_mse": sum_best_mse / total,
        "best_of_8_obs_mae": sum_best_mae / total,
        "top3_oracle_obs_mse": sum_top3_mse / total,
        "top3_oracle_obs_mae": sum_top3_mae / total,
        "selected_obs_mse": sum_selected_mse / total,
        "selected_obs_mae": sum_selected_mae / total,
        "fused_obs_mse": sum_fused_obs_mse / total,
        "fused_obs_mae": sum_fused_obs_mae / total,
        "fused_latent_decode_obs_mse": sum_fused_latent_mse / total,
        "fused_latent_decode_obs_mae": sum_fused_latent_mae / total,
        "branch_diversity_obs_mse": sum_diversity / total,
        "score_entropy": sum_entropy / total,
        "score_max_prob": sum_max_prob / total,
        "winner_counts": winner_counts.tolist(),
        "selected_counts": selected_counts.tolist(),
    }

    prefix = os.path.join(args.output_dir, flag)
    with open(prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.save_predictions:
        np.save(prefix + "_y_branches.npy", np.concatenate(y_branches_all, axis=0))
        np.save(prefix + "_y_selected.npy", np.concatenate(y_selected_all, axis=0))
        np.save(prefix + "_y_fused_obs.npy", np.concatenate(y_fused_obs_all, axis=0))
        np.save(prefix + "_y_fused_latent_decode.npy", np.concatenate(y_fused_latent_all, axis=0))
        np.save(prefix + "_y_true.npy", np.concatenate(y_true_all, axis=0))
        np.save(prefix + "_scores.npy", np.concatenate(scores_all, axis=0))
        np.save(prefix + "_probs.npy", np.concatenate(probs_all, axis=0))
        np.save(prefix + "_selected_index.npy", np.concatenate(selected_all, axis=0))

    print(f"\nDecoded multi-branch evaluation [{flag}]")
    print(f"  samples: {total}")
    print(f"  best-of-{num_branches} obs MSE/MAE: {metrics['best_of_8_obs_mse']:.6f} / {metrics['best_of_8_obs_mae']:.6f}")
    print(f"  top-{metrics['top_k']} oracle obs MSE/MAE: {metrics['top3_oracle_obs_mse']:.6f} / {metrics['top3_oracle_obs_mae']:.6f}")
    print(f"  selected obs MSE/MAE: {metrics['selected_obs_mse']:.6f} / {metrics['selected_obs_mae']:.6f}")
    print(f"  fused obs MSE/MAE: {metrics['fused_obs_mse']:.6f} / {metrics['fused_obs_mae']:.6f}")
    print(f"  D(fused latent) obs MSE/MAE: {metrics['fused_latent_decode_obs_mse']:.6f} / {metrics['fused_latent_decode_obs_mae']:.6f}")
    print(f"  branch diversity obs MSE: {metrics['branch_diversity_obs_mse']:.6f}")
    print(f"  score entropy / max prob: {metrics['score_entropy']:.4f} / {metrics['score_max_prob']:.4f}")
    print(f"  winner counts: {metrics['winner_counts']}")
    print(f"  selected counts: {metrics['selected_counts']}")
    print(f"  saved metrics: {prefix}_metrics.json")
    if args.save_predictions:
        print(f"  saved predictions prefix: {prefix}_*.npy")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Decode multi-branch latent predictions to observation space")
    parser.add_argument("--multibranch_path", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_flags", type=str, default="val,test")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--top_k", type=int, default=3)

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
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_branches", type=int, default=8)
    args = parser.parse_args()

    flags = [flag.strip() for flag in args.eval_flags.split(",") if flag.strip()]
    all_metrics = {flag: evaluate(args, flag) for flag in flags}
    with open(os.path.join(args.output_dir, "metrics_all.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
