import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from my_utils import model_dict


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_flat_zy(z_y):
    flat = z_y.reshape(z_y.shape[0], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / (flat.std(axis=1, keepdims=True) + 1e-6)
    return flat.astype(np.float32)


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


class LatentBackboneMultiBranch(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.d_model = args.d_model
        self.num_branches = args.num_branches

        backbone_args = SimpleNamespace(**vars(args))
        backbone_args.enc_in = args.d_model
        backbone_args.dec_in = args.d_model
        backbone_args.c_out = args.d_model
        self.backbone = model_dict[args.model].Model(backbone_args).float()

        self.branch_delta = torch.nn.Sequential(
            torch.nn.LayerNorm(args.d_model),
            torch.nn.Linear(args.d_model, args.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.hidden_dim, args.num_branches * args.d_model),
        )
        self.context_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(args.d_model),
            torch.nn.Linear(args.d_model, args.hidden_dim),
            torch.nn.GELU(),
        )
        self.branch_score_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(args.d_model),
            torch.nn.Linear(args.d_model, args.hidden_dim),
            torch.nn.GELU(),
        )
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(args.hidden_dim * 2),
            torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.hidden_dim, 1),
        )

    def forward_with_base(self, z_x):
        z_base = self.backbone(z_x, None, None, None)
        z_base = z_base[:, -self.pred_len :, :]
        deltas = self.branch_delta(z_base)
        deltas = deltas.view(z_base.size(0), self.pred_len, self.num_branches, self.d_model)
        deltas = deltas.permute(0, 2, 1, 3).contiguous()
        branches = z_base[:, None, :, :] + deltas
        return z_base, branches

    def forward(self, z_x, detach_score_candidates=False):
        z_base, branches = self.forward_with_base(z_x)
        context_hidden = self.context_proj(z_base.mean(dim=1))
        context_hidden = context_hidden[:, None, :].expand(-1, self.num_branches, -1)
        branch_hidden = self.branch_score_proj(branches.mean(dim=2))
        logits = self.scorer(torch.cat([context_hidden, branch_hidden], dim=-1)).squeeze(-1)
        return branches, logits


def official_loader(args, flag):
    data_set, _ = data_provider(args, flag)
    return DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def extract_split(args, autoencoder, device, flag, out_dir):
    loader = official_loader(args, flag)
    z_x_all, z_y_all, y_all = [], [], []

    print(f"\nExtract official {flag} latents...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            y_true = batch_y[:, -args.pred_len :, :].float().to(device)
            z_x = autoencoder.encode(batch_x)
            z_y = autoencoder.encode(y_true)
            z_x_all.append(z_x.cpu().numpy().astype(np.float32))
            z_y_all.append(z_y.cpu().numpy().astype(np.float32))
            y_all.append(y_true.cpu().numpy().astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"  {flag}: extracted {i + 1}/{len(loader)} batches")

    z_x = np.concatenate(z_x_all, axis=0)
    z_y = np.concatenate(z_y_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    np.save(os.path.join(out_dir, f"{flag}_zx.npy"), z_x)
    np.save(os.path.join(out_dir, f"{flag}_zy.npy"), z_y)
    np.save(os.path.join(out_dir, f"{flag}_y.npy"), y)
    print(f"  {flag}_zx: {z_x.shape}")
    print(f"  {flag}_zy: {z_y.shape}")
    print(f"  {flag}_y:  {y.shape}")
    return z_x, z_y, y


def fit_predict_kmeans(args, splits, out_dir):
    train_norm = normalize_flat_zy(splits["train"]["z_y"])
    kmeans = KMeans(
        n_clusters=args.num_branches,
        init="k-means++",
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
        random_state=args.seed,
    )
    train_labels = kmeans.fit_predict(train_norm).astype(np.int64)
    joblib.dump(kmeans, os.path.join(out_dir, f"kmeans_norm_K{args.num_branches}.joblib"))
    np.save(os.path.join(out_dir, f"kmeans_norm_centers_K{args.num_branches}.npy"), kmeans.cluster_centers_)

    labels = {"train": train_labels}
    for flag in ["val", "test"]:
        labels[flag] = kmeans.predict(normalize_flat_zy(splits[flag]["z_y"])).astype(np.int64)

    summary = {}
    for flag, lab in labels.items():
        counts = np.bincount(lab, minlength=args.num_branches)
        summary[flag] = {
            "counts": counts.tolist(),
            "ratios": (counts / max(len(lab), 1)).tolist(),
        }
        np.save(os.path.join(out_dir, f"{flag}_labels_K{args.num_branches}.npy"), lab)

    with open(os.path.join(out_dir, "kmeans_label_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nKMeans labels from official train-fitted prior:")
    print(json.dumps(summary, indent=2))
    return labels, summary


def branch_errors(branches, target):
    return ((branches - target[:, None]) ** 2).mean(dim=(2, 3))


def assigned_mse(branches, labels, target):
    batch_idx = torch.arange(branches.size(0), device=branches.device)
    return F.mse_loss(branches[batch_idx, labels], target)


def best_mse(branches, target):
    return branch_errors(branches, target).min(dim=1).values.mean()


def fused_prediction(branches, logits):
    probs = torch.softmax(logits, dim=1)
    fused = (probs[:, :, None, None] * branches).sum(dim=1)
    return fused, probs


def soft_score_targets(branches, target, tau):
    errors = branch_errors(branches, target)
    return torch.softmax(-errors.detach() / tau, dim=1)


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def pairwise_diversity(x_branches):
    if x_branches.size(1) < 2:
        return torch.tensor(0.0, device=x_branches.device)
    flat = x_branches.flatten(start_dim=2)
    values = []
    for i in range(x_branches.size(1)):
        for j in range(i + 1, x_branches.size(1)):
            values.append(((flat[:, i] - flat[:, j]) ** 2).mean(dim=1))
    return torch.stack(values, dim=1).mean()


def make_tensor_loader(split, labels, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split["z_x"]),
            torch.from_numpy(split["z_y"]),
            torch.from_numpy(split["y"]),
            torch.from_numpy(labels),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def evaluate_latent(model, loader, args, device, autoencoder=None):
    model.eval()
    totals = {
        "loss": 0.0,
        "assign_mse": 0.0,
        "best_mse": 0.0,
        "fuse_mse": 0.0,
        "score_loss": 0.0,
        "selected_mse": 0.0,
        "diversity": 0.0,
        "entropy": 0.0,
        "max_prob": 0.0,
        "winner_at_1": 0.0,
        "winner_at_3": 0.0,
    }
    count = 0
    winner_counts = np.zeros(args.num_branches, dtype=np.int64)

    with torch.no_grad():
        for z_x, z_y, y, labels in loader:
            z_x = z_x.to(device)
            z_y = z_y.to(device)
            labels = labels.to(device)
            branches, logits = model(z_x)
            fused, probs = fused_prediction(branches, logits)
            assign = assigned_mse(branches, labels, z_y)
            best = best_mse(branches, z_y)
            fuse = F.mse_loss(fused, z_y)
            score_targets = soft_score_targets(branches, z_y, args.score_tau)
            score = soft_cross_entropy(logits, score_targets)
            loss = (
                args.assign_weight * assign
                + args.best_weight * best
                + args.fuse_weight * fuse
                + args.score_weight * score
            )
            errors = branch_errors(branches, z_y)
            winners = errors.argmin(dim=1)
            selected = logits.argmax(dim=1)
            top_idx = logits.topk(min(3, args.num_branches), dim=1).indices
            batch_idx = torch.arange(z_x.size(0), device=device)
            selected_loss = errors[batch_idx, selected].mean()
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1).mean()
            max_prob = probs.max(dim=1).values.mean()
            winner_at_1 = (selected == winners).float().mean()
            winner_at_3 = (top_idx == winners[:, None]).any(dim=1).float().mean()

            bsz = z_x.size(0)
            count += bsz
            totals["loss"] += loss.item() * bsz
            totals["assign_mse"] += assign.item() * bsz
            totals["best_mse"] += best.item() * bsz
            totals["fuse_mse"] += fuse.item() * bsz
            totals["score_loss"] += score.item() * bsz
            totals["selected_mse"] += selected_loss.item() * bsz
            totals["diversity"] += pairwise_diversity(branches).item() * bsz
            totals["entropy"] += entropy.item() * bsz
            totals["max_prob"] += max_prob.item() * bsz
            totals["winner_at_1"] += winner_at_1.item() * bsz
            totals["winner_at_3"] += winner_at_3.item() * bsz
            winner_counts += np.bincount(winners.cpu().numpy(), minlength=args.num_branches)

    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["winner_counts"] = winner_counts.tolist()
    return result


def train_multibranch(args, splits, labels, device, out_dir, autoencoder):
    train_loader = make_tensor_loader(splits["train"], labels["train"], args.batch_size, True)
    val_loader = make_tensor_loader(splits["val"], labels["val"], args.batch_size, False)

    model = LatentBackboneMultiBranch(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = os.path.join(out_dir, "best_multibranch_official.pt")
    history = []
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print("\nTrain official LatentTSF-backbone multi-branch predictor...")
    print(f"Backbone: {args.model} in latent space, enc_in/dec_in/c_out = d_model = {args.d_model}")
    print(
        f"Loss = {args.assign_weight} * L_assign + {args.best_weight} * L_best + "
        f"{args.fuse_weight} * L_fuse + {args.score_weight} * L_score"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for z_x, z_y, y, lab in train_loader:
            z_x = z_x.to(device)
            z_y = z_y.to(device)
            lab = lab.to(device)
            branches, logits = model(z_x)
            fused, _ = fused_prediction(branches, logits)
            assign = assigned_mse(branches, lab, z_y)
            best = best_mse(branches, z_y)
            fuse = F.mse_loss(fused, z_y)
            score_targets = soft_score_targets(branches, z_y, args.score_tau)
            score = soft_cross_entropy(logits, score_targets)
            loss = (
                args.assign_weight * assign
                + args.best_weight * best
                + args.fuse_weight * fuse
                + args.score_weight * score
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_eval = evaluate_latent(model, train_loader, args, device, autoencoder)
        val_eval = evaluate_latent(model, val_loader, args, device, autoencoder)
        row = {
            "epoch": epoch,
            "train": train_eval,
            "val": val_eval,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train assign {train_eval['assign_mse']:.5f} best {train_eval['best_mse']:.5f} | "
            f"val assign {val_eval['assign_mse']:.5f} best {val_eval['best_mse']:.5f} "
            f"selected {val_eval['selected_mse']:.5f} "
            f"w@1 {val_eval['winner_at_1']:.3f} w@3 {val_eval['winner_at_3']:.3f} "
            f"div {val_eval['diversity']:.5f}",
            flush=True,
        )

        metric = val_eval[args.early_stop_metric]
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
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_eval = evaluate_latent(model, train_loader, args, device, autoencoder)
    val_eval = evaluate_latent(model, val_loader, args, device, autoencoder)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "early_stop_metric": args.early_stop_metric,
        "train": train_eval,
        "val": val_eval,
        "history": history,
        "best_model_path": best_path,
    }
    with open(os.path.join(out_dir, "official_train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return model, metrics


def decode_eval(args, model, autoencoder, split, labels, flag, device, out_dir):
    loader = make_tensor_loader(split, labels, args.batch_size, False)
    model.eval()
    autoencoder.eval()

    sums = {
        "best_mse": 0.0,
        "best_mae": 0.0,
        "top3_mse": 0.0,
        "top3_mae": 0.0,
        "selected_mse": 0.0,
        "selected_mae": 0.0,
        "fused_obs_mse": 0.0,
        "fused_obs_mae": 0.0,
        "fused_latent_mse": 0.0,
        "fused_latent_mae": 0.0,
        "diversity": 0.0,
        "entropy": 0.0,
        "max_prob": 0.0,
        "winner_at_1": 0.0,
        "winner_at_3": 0.0,
    }
    total = 0
    winner_counts = np.zeros(args.num_branches, dtype=np.int64)
    selected_counts = np.zeros(args.num_branches, dtype=np.int64)
    saved = {"y_branches": [], "y_selected": [], "y_fused_obs": [], "y_true": [], "scores": [], "selected": []}

    with torch.no_grad():
        for z_x, z_y, y_true, lab in loader:
            z_x = z_x.to(device)
            y_true = y_true.to(device)
            z_branches, scores = model(z_x)
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
            top_idx = scores.topk(min(3, args.num_branches), dim=1).indices
            top_mse = torch.gather(mse_by_branch, 1, top_idx)
            top_mae = torch.gather(mae_by_branch, 1, top_idx)
            top_pos = top_mse.argmin(dim=1, keepdim=True)
            winner_at_1 = (selected_idx == winner_idx).float()
            winner_at_3 = (top_idx == winner_idx[:, None]).any(dim=1).float()
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
            max_prob = probs.max(dim=1).values

            bsz = z_x.size(0)
            total += bsz
            sums["best_mse"] += mse_by_branch[batch_idx, winner_idx].sum().item()
            sums["best_mae"] += mae_by_branch[batch_idx, winner_idx].sum().item()
            sums["top3_mse"] += top_mse.gather(1, top_pos).sum().item()
            sums["top3_mae"] += top_mae.gather(1, top_pos).sum().item()
            sums["selected_mse"] += ((y_selected - y_true) ** 2).mean().item() * bsz
            sums["selected_mae"] += (y_selected - y_true).abs().mean().item() * bsz
            sums["fused_obs_mse"] += ((y_fused_obs - y_true) ** 2).mean().item() * bsz
            sums["fused_obs_mae"] += (y_fused_obs - y_true).abs().mean().item() * bsz
            sums["fused_latent_mse"] += ((y_fused_latent - y_true) ** 2).mean().item() * bsz
            sums["fused_latent_mae"] += (y_fused_latent - y_true).abs().mean().item() * bsz
            sums["diversity"] += pairwise_diversity(y_branches).item() * bsz
            sums["entropy"] += entropy.sum().item()
            sums["max_prob"] += max_prob.sum().item()
            sums["winner_at_1"] += winner_at_1.sum().item()
            sums["winner_at_3"] += winner_at_3.sum().item()
            winner_counts += np.bincount(winner_idx.cpu().numpy(), minlength=args.num_branches)
            selected_counts += np.bincount(selected_idx.cpu().numpy(), minlength=args.num_branches)

            if args.save_predictions:
                saved["y_branches"].append(y_branches.cpu().numpy().astype(np.float32))
                saved["y_selected"].append(y_selected.cpu().numpy().astype(np.float32))
                saved["y_fused_obs"].append(y_fused_obs.cpu().numpy().astype(np.float32))
                saved["y_true"].append(y_true.cpu().numpy().astype(np.float32))
                saved["scores"].append(scores.cpu().numpy().astype(np.float32))
                saved["selected"].append(selected_idx.cpu().numpy().astype(np.int64))

    metrics = {
        "flag": flag,
        "num_samples": total,
        "best_of_8_obs_mse": sums["best_mse"] / total,
        "best_of_8_obs_mae": sums["best_mae"] / total,
        "top3_oracle_obs_mse": sums["top3_mse"] / total,
        "top3_oracle_obs_mae": sums["top3_mae"] / total,
        "selected_obs_mse": sums["selected_mse"] / total,
        "selected_obs_mae": sums["selected_mae"] / total,
        "fused_obs_mse": sums["fused_obs_mse"] / total,
        "fused_obs_mae": sums["fused_obs_mae"] / total,
        "fused_latent_decode_obs_mse": sums["fused_latent_mse"] / total,
        "fused_latent_decode_obs_mae": sums["fused_latent_mae"] / total,
        "branch_diversity_obs_mse": sums["diversity"] / total,
        "score_entropy": sums["entropy"] / total,
        "score_max_prob": sums["max_prob"] / total,
        "scorer_winner_at_1": sums["winner_at_1"] / total,
        "scorer_winner_at_3": sums["winner_at_3"] / total,
        "winner_counts": winner_counts.tolist(),
        "selected_counts": selected_counts.tolist(),
    }
    with open(os.path.join(out_dir, f"{flag}_decoded_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if args.save_predictions:
        for key, chunks in saved.items():
            np.save(os.path.join(out_dir, f"{flag}_{key}.npy"), np.concatenate(chunks, axis=0))

    print(f"\nOfficial decoded evaluation [{flag}]")
    print(f"  best-of-8 obs MSE/MAE: {metrics['best_of_8_obs_mse']:.6f} / {metrics['best_of_8_obs_mae']:.6f}")
    print(f"  selected obs MSE/MAE: {metrics['selected_obs_mse']:.6f} / {metrics['selected_obs_mae']:.6f}")
    print(f"  fused obs MSE/MAE: {metrics['fused_obs_mse']:.6f} / {metrics['fused_obs_mae']:.6f}")
    print(f"  scorer winner@1/@3: {metrics['scorer_winner_at_1']:.4f} / {metrics['scorer_winner_at_3']:.4f}")
    print(f"  winner counts: {metrics['winner_counts']}")
    print(f"  selected counts: {metrics['selected_counts']}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Official split multi-branch latent pipeline")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/official_multibranch_ETTh1_sl96_pl96")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--model", type=str, default="DLinear")
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
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--num_branches", type=int, default=8)
    parser.add_argument("--kmeans_n_init", type=int, default=20)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--assign_weight", type=float, default=1.0)
    parser.add_argument("--best_weight", type=float, default=0.1)
    parser.add_argument("--fuse_weight", type=float, default=0.3)
    parser.add_argument("--score_weight", type=float, default=0.1)
    parser.add_argument("--score_tau", type=float, default=0.05)
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="assign_mse",
        choices=["loss", "assign_mse", "best_mse", "fuse_mse", "selected_mse"],
    )
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    splits = {}
    for flag in ["train", "val", "test"]:
        z_x, z_y, y = extract_split(args, autoencoder, device, flag, args.output_dir)
        splits[flag] = {"z_x": z_x, "z_y": z_y, "y": y}

    labels, label_summary = fit_predict_kmeans(args, splits, args.output_dir)
    model, train_metrics = train_multibranch(args, splits, labels, device, args.output_dir, autoencoder)
    val_decoded = decode_eval(args, model, autoencoder, splits["val"], labels["val"], "val", device, args.output_dir)
    test_decoded = decode_eval(args, model, autoencoder, splits["test"], labels["test"], "test", device, args.output_dir)
    summary = {
        "label_summary": label_summary,
        "train_metrics": train_metrics,
        "val_decoded": val_decoded,
        "test_decoded": test_decoded,
    }
    with open(os.path.join(args.output_dir, "official_pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
