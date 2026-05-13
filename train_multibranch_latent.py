import argparse
import json
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MultiBranchLatentPredictor(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        num_branches,
        hidden_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_branches = num_branches

        input_dim = seq_len * d_model
        output_dim = num_branches * seq_len * d_model

        self.backbone = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branch_head = nn.Linear(hidden_dim, output_dim)

        self.branch_score_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_x):
        hidden = self.backbone(z_x)
        branches = self.branch_head(hidden).view(
            z_x.size(0), self.num_branches, self.seq_len, self.d_model
        )
        branch_hidden = self.branch_score_proj(branches.mean(dim=2))
        context_hidden = hidden[:, None, :].expand(-1, self.num_branches, -1)
        logits = self.scorer(torch.cat([context_hidden, branch_hidden], dim=-1)).squeeze(-1)
        return branches, logits


class MinimalMLPAutoEncoder(nn.Module):
    def __init__(self, enc_in, d_model, d_ff):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, enc_in),
        )

    def decode(self, latent):
        return self.decoder(latent)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def topk_accuracy(logits, labels, k):
    topk = logits.topk(k, dim=1).indices
    return (topk == labels[:, None]).any(dim=1).float().mean().item()


def assigned_branch_mse(branches, labels, target):
    batch_idx = torch.arange(branches.size(0), device=branches.device)
    assigned = branches[batch_idx, labels]
    return F.mse_loss(assigned, target)


def best_branch_mse(branches, target):
    mse_all = ((branches - target[:, None]) ** 2).mean(dim=(2, 3))
    return mse_all.min(dim=1).values.mean()


def branch_errors(branches, target):
    return ((branches - target[:, None]) ** 2).mean(dim=(2, 3))


def soft_score_targets(branches, target, tau):
    errors = branch_errors(branches, target)
    winners = errors.argmin(dim=1)
    targets = torch.softmax(-errors.detach() / tau, dim=1)
    return targets, winners, errors


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def score_loss(branches, logits, labels, target, score_target, tau):
    if score_target == "cluster":
        return F.cross_entropy(logits, labels)
    soft_targets, _, _ = soft_score_targets(branches, target, tau)
    return soft_cross_entropy(logits, soft_targets)


def fused_prediction(branches, logits, detach_logits=False):
    if detach_logits:
        logits = logits.detach()
    probs = torch.softmax(logits, dim=1)
    fused = (probs[:, :, None, None] * branches).sum(dim=1)
    return fused, probs


def branch_diversity(branches):
    if branches.size(1) < 2:
        return 0.0

    flat = branches.flatten(start_dim=2)
    distances = []
    for i in range(branches.size(1)):
        for j in range(i + 1, branches.size(1)):
            distances.append(F.mse_loss(flat[:, i], flat[:, j], reduction="none").mean(dim=1))
    return torch.stack(distances, dim=1).mean().item()


def load_decoder(args, device):
    if args.autoencoder_path is None:
        return None

    decoder = MinimalMLPAutoEncoder(
        enc_in=args.enc_in,
        d_model=args.d_model,
        d_ff=args.d_ff,
    ).to(device)
    state_dict = torch.load(args.autoencoder_path, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    decoder.load_state_dict(state_dict)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


def per_branch_assigned_mse(model, loader, device, num_branches):
    model.eval()
    sums = np.zeros(num_branches, dtype=np.float64)
    counts = np.zeros(num_branches, dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            z_x, z_y, labels = batch[:3]
            z_x = z_x.to(device)
            z_y = z_y.to(device)
            labels = labels.to(device)
            branches, _ = model(z_x)
            batch_idx = torch.arange(branches.size(0), device=device)
            assigned = branches[batch_idx, labels]
            sample_mse = ((assigned - z_y) ** 2).mean(dim=(1, 2)).cpu().numpy()
            labels_np = labels.cpu().numpy()
            for k in range(num_branches):
                mask = labels_np == k
                if mask.any():
                    sums[k] += sample_mse[mask].sum()
                    counts[k] += mask.sum()

    return np.divide(
        sums,
        counts,
        out=np.full(num_branches, np.nan, dtype=np.float64),
        where=counts != 0,
    )


def evaluate(
    model,
    loader,
    assign_weight,
    ce_weight,
    best_weight,
    fuse_weight,
    score_target,
    score_tau,
    detach_fuse_score,
    device,
    num_branches,
    decoder=None,
):
    model.eval()
    total_loss = 0.0
    total_assign = 0.0
    total_best = 0.0
    total_fuse = 0.0
    total_gate = 0.0
    total_diversity = 0.0
    total_gate_selected_mse = 0.0
    total_gate_entropy = 0.0
    total_gate_max_prob = 0.0
    total_count = 0
    winner_counts = np.zeros(num_branches, dtype=np.int64)
    logits_all = []
    labels_all = []
    obs_sums = {
        "assigned_obs_mse": 0.0,
        "assigned_obs_mae": 0.0,
        "best_obs_mse": 0.0,
        "best_obs_mae": 0.0,
        "gate_obs_mse": 0.0,
        "gate_obs_mae": 0.0,
        "fuse_obs_mse": 0.0,
        "fuse_obs_mae": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                z_x, z_y, labels, y_obs = batch
            else:
                z_x, z_y, labels = batch
                y_obs = None
            z_x = z_x.to(device)
            z_y = z_y.to(device)
            labels = labels.to(device)
            if y_obs is not None:
                y_obs = y_obs.to(device)

            branches, logits = model(z_x)
            fused, probs = fused_prediction(branches, logits)
            fused_loss_pred, _ = fused_prediction(branches, logits, detach_fuse_score)
            soft_targets, target_winners, _ = soft_score_targets(branches, z_y, score_tau)
            assign_loss = assigned_branch_mse(branches, labels, z_y)
            best_loss = best_branch_mse(branches, z_y)
            fuse_loss = F.mse_loss(fused_loss_pred, z_y)
            gate_loss = (
                F.cross_entropy(logits, labels)
                if score_target == "cluster"
                else soft_cross_entropy(logits, soft_targets)
            )
            loss = (
                assign_weight * assign_loss
                + best_weight * best_loss
                + fuse_weight * fuse_loss
                + ce_weight * gate_loss
            )
            batch_idx = torch.arange(branches.size(0), device=device)
            pred = logits.argmax(dim=1)
            mse_all = ((branches - z_y[:, None]) ** 2).mean(dim=(2, 3))
            winners = target_winners.cpu().numpy()
            winner_counts += np.bincount(winners, minlength=num_branches)
            gate_selected_mse = mse_all[batch_idx, pred].mean()
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1).mean()
            max_prob = probs.max(dim=1).values.mean()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_assign += assign_loss.item() * batch_size
            total_best += best_loss.item() * batch_size
            total_fuse += fuse_loss.item() * batch_size
            total_gate += gate_loss.item() * batch_size
            total_diversity += branch_diversity(branches) * batch_size
            total_gate_selected_mse += gate_selected_mse.item() * batch_size
            total_gate_entropy += entropy.item() * batch_size
            total_gate_max_prob += max_prob.item() * batch_size
            total_count += batch_size
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())

            if decoder is not None and y_obs is not None:
                decoded = decoder.decode(branches)
                assigned_obs = decoded[batch_idx, labels]
                gate_obs = decoded[batch_idx, pred]
                fuse_obs = decoder.decode(fused)
                obs_mse_all = ((decoded - y_obs[:, None]) ** 2).mean(dim=(2, 3))
                best_obs_idx = obs_mse_all.argmin(dim=1)
                best_obs = decoded[batch_idx, best_obs_idx]

                obs_sums["assigned_obs_mse"] += ((assigned_obs - y_obs) ** 2).mean().item() * batch_size
                obs_sums["assigned_obs_mae"] += (assigned_obs - y_obs).abs().mean().item() * batch_size
                obs_sums["best_obs_mse"] += ((best_obs - y_obs) ** 2).mean().item() * batch_size
                obs_sums["best_obs_mae"] += (best_obs - y_obs).abs().mean().item() * batch_size
                obs_sums["gate_obs_mse"] += ((gate_obs - y_obs) ** 2).mean().item() * batch_size
                obs_sums["gate_obs_mae"] += (gate_obs - y_obs).abs().mean().item() * batch_size
                obs_sums["fuse_obs_mse"] += ((fuse_obs - y_obs) ** 2).mean().item() * batch_size
                obs_sums["fuse_obs_mae"] += (fuse_obs - y_obs).abs().mean().item() * batch_size

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    pred = logits_all.argmax(dim=1)
    winner_labels = []
    with torch.no_grad():
        for batch in loader:
            z_x, z_y = batch[0].to(device), batch[1].to(device)
            branches, _ = model(z_x)
            winner_labels.append(branch_errors(branches, z_y).argmin(dim=1).cpu())
    winner_labels = torch.cat(winner_labels, dim=0)

    result = {
        "loss": total_loss / max(total_count, 1),
        "assign_mse": total_assign / max(total_count, 1),
        "best_mse": total_best / max(total_count, 1),
        "fuse_mse": total_fuse / max(total_count, 1),
        "gate_selected_mse": total_gate_selected_mse / max(total_count, 1),
        "gate_ce": total_gate / max(total_count, 1),
        "gate_entropy": total_gate_entropy / max(total_count, 1),
        "gate_max_prob": total_gate_max_prob / max(total_count, 1),
        "branch_diversity": total_diversity / max(total_count, 1),
        "winner_counts": winner_counts.tolist(),
        "gate_top1": (pred == labels_all).float().mean().item(),
        "gate_top3": topk_accuracy(logits_all, labels_all, min(3, num_branches)),
        "score_winner_top1": (pred == winner_labels).float().mean().item(),
        "score_winner_top3": topk_accuracy(logits_all, winner_labels, min(3, num_branches)),
    }
    if decoder is not None:
        result.update({key: value / max(total_count, 1) for key, value in obs_sums.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description="Minimal multi-branch latent predictor")
    parser.add_argument(
        "--latent_dir",
        type=str,
        default="./latent_outputs/ETTh1_sl96_pl96_dm32_dff64_MLP",
    )
    parser.add_argument("--zx_file", type=str, default="train_zx.npy")
    parser.add_argument("--zy_file", type=str, default="train_zy.npy")
    parser.add_argument("--label_file", type=str, default="zy_kmeanspp_labels_K8.npy")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_branches", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--assign_weight", type=float, default=1.0)
    parser.add_argument("--gate_weight", type=float, default=0.1)
    parser.add_argument("--best_weight", type=float, default=0.0)
    parser.add_argument("--fuse_weight", type=float, default=0.0)
    parser.add_argument("--score_target", type=str, default="cluster", choices=["cluster", "error"])
    parser.add_argument("--score_tau", type=float, default=0.05)
    parser.add_argument(
        "--detach_fuse_score",
        action="store_true",
        help="Stop L_fuse from updating the scoring head; useful for TNT-style candidate scoring.",
    )
    parser.add_argument("--autoencoder_path", type=str, default=None)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    default_name = (
        "multibranch_k8_minimal"
        if args.assign_weight == 1 and args.best_weight == 0 and args.fuse_weight == 0
        else (
            f"multibranch_k8_assign{args.assign_weight:g}_best{args.best_weight:g}_"
            f"fuse{args.fuse_weight:g}_{args.score_target}"
        )
    )
    if args.detach_fuse_score:
        default_name += "_detachfuse"
    out_dir = args.out_dir or os.path.join(args.latent_dir, default_name)
    os.makedirs(out_dir, exist_ok=True)

    z_x = np.load(os.path.join(args.latent_dir, args.zx_file)).astype(np.float32)
    z_y = np.load(os.path.join(args.latent_dir, args.zy_file)).astype(np.float32)
    y_obs_path = os.path.join(args.latent_dir, "train_y.npy")
    y_obs = np.load(y_obs_path).astype(np.float32) if os.path.exists(y_obs_path) else None
    labels = np.load(os.path.join(args.latent_dir, args.label_file)).astype(np.int64)

    if z_x.shape != z_y.shape:
        raise ValueError(f"Expected z_x and z_y to have same shape, got {z_x.shape} and {z_y.shape}")
    if z_x.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatched sample count: z_x={z_x.shape}, labels={labels.shape}")

    n_samples, seq_len, d_model = z_x.shape
    split = int(n_samples * args.train_ratio)
    train_x, val_x = z_x[:split], z_x[split:]
    train_y, val_y = z_y[:split], z_y[split:]
    train_obs = y_obs[:split] if y_obs is not None else None
    val_obs = y_obs[split:] if y_obs is not None else None
    train_c, val_c = labels[:split], labels[split:]

    train_counts = np.bincount(train_c, minlength=args.num_branches)
    val_counts = np.bincount(val_c, minlength=args.num_branches)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_tensors = [
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
            torch.from_numpy(train_c),
    ]
    val_tensors = [
            torch.from_numpy(val_x),
            torch.from_numpy(val_y),
            torch.from_numpy(val_c),
    ]
    if train_obs is not None:
        train_tensors.append(torch.from_numpy(train_obs))
        val_tensors.append(torch.from_numpy(val_obs))

    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(*val_tensors),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MultiBranchLatentPredictor(
        seq_len=seq_len,
        d_model=d_model,
        num_branches=args.num_branches,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    decoder = load_decoder(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = os.path.join(out_dir, "best_multibranch.pt")

    print(f"Loaded z_x: {z_x.shape}")
    print(f"Loaded z_y: {z_y.shape}")
    print(f"Loaded y_obs: {None if y_obs is None else y_obs.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Train/val split: {split}/{n_samples - split}")
    print(f"Train counts: {train_counts.tolist()}")
    print(f"Val counts: {val_counts.tolist()}")
    print(
        f"Loss: L = {args.assign_weight} * L_assign + {args.best_weight} * L_best + "
        f"{args.fuse_weight} * L_fuse + {args.gate_weight} * L_score"
    )
    print(f"Score target: {args.score_target}, tau={args.score_tau}")
    print(f"Detach score in L_fuse: {args.detach_fuse_score}")
    print(f"Observation decoder: {'enabled' if decoder is not None and y_obs is not None else 'disabled'}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_assign = 0.0
        total_gate = 0.0
        total_count = 0

        for batch in train_loader:
            batch_x, batch_y, batch_c = batch[:3]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_c = batch_c.to(device)

            branches, logits = model(batch_x)
            fused, _ = fused_prediction(branches, logits, args.detach_fuse_score)
            assign_loss = assigned_branch_mse(branches, batch_c, batch_y)
            best_loss = best_branch_mse(branches, batch_y)
            fuse_loss = F.mse_loss(fused, batch_y)
            gate_loss = score_loss(
                branches, logits, batch_c, batch_y, args.score_target, args.score_tau
            )
            loss = (
                args.assign_weight * assign_loss
                + args.best_weight * best_loss
                + args.fuse_weight * fuse_loss
                + args.gate_weight * gate_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = batch_c.size(0)
            total_loss += loss.item() * batch_size
            total_assign += assign_loss.item() * batch_size
            total_gate += gate_loss.item() * batch_size
            total_count += batch_size

        train_eval = evaluate(
            model, train_loader, args.assign_weight, args.gate_weight, args.best_weight, args.fuse_weight,
            args.score_target, args.score_tau, args.detach_fuse_score, device, args.num_branches, decoder
        )
        val_eval = evaluate(
            model, val_loader, args.assign_weight, args.gate_weight, args.best_weight, args.fuse_weight,
            args.score_target, args.score_tau, args.detach_fuse_score, device, args.num_branches, decoder
        )

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_assign_mse": total_assign / max(total_count, 1),
            "train_gate_ce": total_gate / max(total_count, 1),
            "train_best_mse": train_eval["best_mse"],
            "train_fuse_mse": train_eval["fuse_mse"],
            "train_gate_top1": train_eval["gate_top1"],
            "train_gate_top3": train_eval["gate_top3"],
            "train_branch_diversity": train_eval["branch_diversity"],
            "val_loss": val_eval["loss"],
            "val_assign_mse": val_eval["assign_mse"],
            "val_best_mse": val_eval["best_mse"],
            "val_fuse_mse": val_eval["fuse_mse"],
            "val_gate_selected_mse": val_eval["gate_selected_mse"],
            "val_gate_ce": val_eval["gate_ce"],
            "val_gate_top1": val_eval["gate_top1"],
            "val_gate_top3": val_eval["gate_top3"],
            "val_score_winner_top1": val_eval["score_winner_top1"],
            "val_score_winner_top3": val_eval["score_winner_top3"],
            "val_gate_entropy": val_eval["gate_entropy"],
            "val_gate_max_prob": val_eval["gate_max_prob"],
            "val_branch_diversity": val_eval["branch_diversity"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train assign {row['train_assign_mse']:.5f} best {row['train_best_mse']:.5f} gate@1 {row['train_gate_top1']:.4f} | "
            f"val assign {row['val_assign_mse']:.5f} best {row['val_best_mse']:.5f} fuse {row['val_fuse_mse']:.5f} "
            f"label@1 {row['val_gate_top1']:.4f} win@1 {row['val_score_winner_top1']:.4f} "
            f"win@3 {row['val_score_winner_top3']:.4f} "
            f"H {row['val_gate_entropy']:.3f} maxp {row['val_gate_max_prob']:.3f} div {row['val_branch_diversity']:.5f}",
            flush=True,
        )

        stop_metric_name = "assign_mse" if args.assign_weight > 0 else "best_mse"
        stop_metric = val_eval[stop_metric_name]
        if stop_metric < best_metric:
            best_metric = stop_metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    f"val_{stop_metric_name}": best_metric,
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

    train_eval = evaluate(
        model, train_loader, args.assign_weight, args.gate_weight, args.best_weight, args.fuse_weight,
        args.score_target, args.score_tau, args.detach_fuse_score, device, args.num_branches, decoder
    )
    val_eval = evaluate(
        model, val_loader, args.assign_weight, args.gate_weight, args.best_weight, args.fuse_weight,
        args.score_target, args.score_tau, args.detach_fuse_score, device, args.num_branches, decoder
    )
    train_branch_mse = per_branch_assigned_mse(model, train_loader, device, args.num_branches)
    val_branch_mse = per_branch_assigned_mse(model, val_loader, device, args.num_branches)

    metrics = {
        "best_epoch": best_epoch,
        "assign_weight": args.assign_weight,
        "gate_weight": args.gate_weight,
        "best_weight": args.best_weight,
        "fuse_weight": args.fuse_weight,
        "score_target": args.score_target,
        "score_tau": args.score_tau,
        "detach_fuse_score": args.detach_fuse_score,
        "train_counts": train_counts.tolist(),
        "val_counts": val_counts.tolist(),
        "train": {
            **{k: v for k, v in train_eval.items() if k != "confusion_matrix"},
            "per_branch_assigned_mse": train_branch_mse.tolist(),
        },
        "val": {
            **{k: v for k, v in val_eval.items() if k != "confusion_matrix"},
            "per_branch_assigned_mse": val_branch_mse.tolist(),
        },
        "history": history,
        "best_model_path": best_path,
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nBest multi-branch metrics")
    print(f"  best_epoch: {best_epoch}")
    print(f"  val assign MSE: {metrics['val']['assign_mse']:.6f}")
    print(f"  val best-of-8 latent MSE: {metrics['val']['best_mse']:.6f}")
    print(f"  val fused latent MSE: {metrics['val']['fuse_mse']:.6f}")
    print(f"  val gate-selected latent MSE: {metrics['val']['gate_selected_mse']:.6f}")
    print(f"  val gate top-1: {metrics['val']['gate_top1']:.4f}")
    print(f"  val gate top-3: {metrics['val']['gate_top3']:.4f}")
    print(f"  val score winner top-1/top-3: {metrics['val']['score_winner_top1']:.4f} / {metrics['val']['score_winner_top3']:.4f}")
    print(f"  val gate entropy / max prob: {metrics['val']['gate_entropy']:.4f} / {metrics['val']['gate_max_prob']:.4f}")
    print(f"  val branch diversity: {metrics['val']['branch_diversity']:.6f}")
    print(f"  val winner counts: {metrics['val']['winner_counts']}")
    if "best_obs_mse" in metrics["val"]:
        print(f"  val best-of-8 obs MSE/MAE: {metrics['val']['best_obs_mse']:.6f} / {metrics['val']['best_obs_mae']:.6f}")
        print(f"  val fused obs MSE/MAE: {metrics['val']['fuse_obs_mse']:.6f} / {metrics['val']['fuse_obs_mae']:.6f}")
        print(f"  val gate-selected obs MSE/MAE: {metrics['val']['gate_obs_mse']:.6f} / {metrics['val']['gate_obs_mae']:.6f}")
        print(f"  val assigned obs MSE/MAE: {metrics['val']['assigned_obs_mse']:.6f} / {metrics['val']['assigned_obs_mae']:.6f}")
    print(f"  val per-branch assigned MSE: {[round(x, 6) for x in metrics['val']['per_branch_assigned_mse']]}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved model: {best_path}")


if __name__ == "__main__":
    main()
