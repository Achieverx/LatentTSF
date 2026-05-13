import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from my_AE import get_autoencoder
from official_multibranch_pipeline import LatentBackboneMultiBranch, pairwise_diversity, set_seed


class SparseResidualGate(torch.nn.Module):
    def __init__(self, d_model, hidden_dim, num_branches, dropout=0.1, alpha_init_bias=-3.0):
        super().__init__()
        self.num_branches = num_branches
        self.score = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model * 4),
            torch.nn.Linear(d_model * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.alpha = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model * 2),
            torch.nn.Linear(d_model * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.constant_(self.alpha[-1].bias, alpha_init_bias)

    def forward(self, z_x, z_base, z_branches, top_m, tau):
        pooled_x = z_x.mean(dim=1)
        pooled_base = z_base.mean(dim=1)
        pooled_x_k = pooled_x[:, None, :].expand(-1, self.num_branches, -1)
        pooled_base_k = pooled_base[:, None, :].expand(-1, self.num_branches, -1)
        pooled_branch = z_branches.mean(dim=2)
        pooled_delta = (z_branches - z_base[:, None, :, :]).mean(dim=2)
        features = torch.cat([pooled_x_k, pooled_base_k, pooled_branch, pooled_delta], dim=-1)
        scores = self.score(features).squeeze(-1)

        if top_m <= 0 or top_m >= self.num_branches:
            weights = torch.softmax(scores / tau, dim=1)
            top_idx = torch.arange(self.num_branches, device=scores.device)[None].expand(scores.size(0), -1)
            z_sparse = (weights[:, :, None, None] * z_branches).sum(dim=1)
        else:
            top_scores, top_idx = scores.topk(top_m, dim=1)
            weights = torch.softmax(top_scores / tau, dim=1)
            gather_idx = top_idx[:, :, None, None].expand(-1, -1, z_branches.size(2), z_branches.size(3))
            selected = torch.gather(z_branches, 1, gather_idx)
            z_sparse = (weights[:, :, None, None] * selected).sum(dim=1)

        alpha = torch.sigmoid(self.alpha(torch.cat([pooled_x, pooled_base], dim=-1))).view(-1, 1, 1)
        z_final = z_base + alpha * (z_sparse - z_base)
        return z_final, scores, weights, top_idx, alpha.squeeze(-1).squeeze(-1)


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


def load_branch_model(args, checkpoint_path, device):
    model = LatentBackboneMultiBranch(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def make_loader(source_dir, flag, batch_size, shuffle):
    z_x = np.load(os.path.join(source_dir, f"{flag}_zx.npy"))
    y = np.load(os.path.join(source_dir, f"{flag}_y.npy"))
    return DataLoader(
        TensorDataset(torch.from_numpy(z_x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def decode_branches(autoencoder, z_branches, enc_in):
    bsz, num_branches, steps, dim = z_branches.shape
    y_branches = autoencoder.decode(z_branches.reshape(bsz * num_branches, steps, dim))
    return y_branches.reshape(bsz, num_branches, steps, enc_in)


def evaluate(args, branch_model, gate, autoencoder, loader, device):
    gate.eval()
    sums = {
        "mse": 0.0,
        "mae": 0.0,
        "base_mse": 0.0,
        "base_mae": 0.0,
        "best_mse": 0.0,
        "best_mae": 0.0,
        "topm_oracle_mse": 0.0,
        "topm_oracle_mae": 0.0,
        "entropy": 0.0,
        "max_prob": 0.0,
        "alpha": 0.0,
        "residual_mse": 0.0,
        "diversity": 0.0,
    }
    top_counts = np.zeros(args.num_branches, dtype=np.int64)
    winner_counts = np.zeros(args.num_branches, dtype=np.int64)
    total = 0

    with torch.no_grad():
        for z_x, y in loader:
            z_x = z_x.float().to(device)
            y = y.float().to(device)
            z_base, z_branches = branch_model.forward_with_base(z_x)
            z_branches = z_branches.detach()
            z_base = z_base.detach()
            z_final, scores, weights, top_idx, alpha = gate(z_x, z_base, z_branches, args.top_m, args.moe_tau)
            y_base = autoencoder.decode(z_base)
            y_branches = decode_branches(autoencoder, z_branches, args.enc_in)
            y_pred = autoencoder.decode(z_final)

            mse_by_branch = ((y_branches - y[:, None]) ** 2).mean(dim=(2, 3))
            mae_by_branch = (y_branches - y[:, None]).abs().mean(dim=(2, 3))
            winner = mse_by_branch.argmin(dim=1)
            top_mse = torch.gather(mse_by_branch, 1, top_idx)
            top_mae = torch.gather(mae_by_branch, 1, top_idx)
            top_pos = top_mse.argmin(dim=1, keepdim=True)

            bsz = z_x.size(0)
            total += bsz
            sums["mse"] += ((y_pred - y) ** 2).mean().item() * bsz
            sums["mae"] += (y_pred - y).abs().mean().item() * bsz
            sums["base_mse"] += ((y_base - y) ** 2).mean().item() * bsz
            sums["base_mae"] += (y_base - y).abs().mean().item() * bsz
            sums["best_mse"] += mse_by_branch[torch.arange(bsz, device=device), winner].sum().item()
            sums["best_mae"] += mae_by_branch[torch.arange(bsz, device=device), winner].sum().item()
            sums["topm_oracle_mse"] += top_mse.gather(1, top_pos).sum().item()
            sums["topm_oracle_mae"] += top_mae.gather(1, top_pos).sum().item()
            sums["entropy"] += (-(weights * (weights + 1e-12).log()).sum(dim=1)).sum().item()
            sums["max_prob"] += weights.max(dim=1).values.sum().item()
            sums["alpha"] += alpha.sum().item()
            sums["residual_mse"] += ((z_final - z_base) ** 2).mean().item() * bsz
            sums["diversity"] += pairwise_diversity(y_branches).item() * bsz
            top_counts += np.bincount(top_idx.cpu().numpy().reshape(-1), minlength=args.num_branches)
            winner_counts += np.bincount(winner.cpu().numpy(), minlength=args.num_branches)

    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    metrics["top_counts"] = top_counts.tolist()
    metrics["winner_counts"] = winner_counts.tolist()
    return metrics


def train_one(args, branch_args, device):
    out_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    autoencoder = load_autoencoder(branch_args, device)
    branch_model = load_branch_model(
        branch_args,
        os.path.join(args.source_dir, "best_multibranch_official.pt"),
        device,
    )
    train_loader = make_loader(args.source_dir, "train", args.batch_size, True)
    val_loader = make_loader(args.source_dir, "val", args.batch_size, False)
    test_loader = make_loader(args.source_dir, "test", args.batch_size, False)

    gate = SparseResidualGate(
        branch_args.d_model,
        args.hidden_dim,
        branch_args.num_branches,
        args.dropout,
        args.alpha_init_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = os.path.join(out_dir, "best_sparse_moe_gate.pt")

    for epoch in range(1, args.epochs + 1):
        gate.train()
        train_loss = 0.0
        seen = 0
        for z_x, y in train_loader:
            z_x = z_x.float().to(device)
            y = y.float().to(device)
            with torch.no_grad():
                z_base, z_branches = branch_model.forward_with_base(z_x)
                z_base = z_base.detach()
                z_branches = z_branches.detach()
            z_final, _, _, _, alpha = gate(z_x, z_base, z_branches, args.top_m, args.moe_tau)
            y_pred = autoencoder.decode(z_final)
            res_loss = ((z_final - z_base) ** 2).mean()
            moe_loss = F.mse_loss(y_pred, y)
            alpha_loss = alpha.mean()
            loss = moe_loss + args.alpha_reg * alpha_loss + args.res_reg * res_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * z_x.size(0)
            seen += z_x.size(0)

        val_metrics = evaluate(args, branch_model, gate, autoencoder, val_loader, device)
        row = {"epoch": epoch, "train_mse": train_loss / max(seen, 1), "val": val_metrics}
        history.append(row)
        print(
            f"{args.run_name} epoch {epoch:03d} | "
            f"train {row['train_mse']:.6f} val moe {val_metrics['mse']:.6f} "
            f"base {val_metrics['base_mse']:.6f} best {val_metrics['best_mse']:.6f} "
            f"alpha {val_metrics['alpha']:.3f} res {val_metrics['residual_mse']:.6f}",
            flush=True,
        )

        if val_metrics["mse"] < best_metric:
            best_metric = val_metrics["mse"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "gate_state_dict": gate.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_mse": best_metric,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"{args.run_name} early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    gate.load_state_dict(checkpoint["gate_state_dict"])
    val_metrics = evaluate(args, branch_model, gate, autoencoder, val_loader, device)
    test_metrics = evaluate(args, branch_model, gate, autoencoder, test_loader, device)
    summary = {
        "top_m": args.top_m,
        "alpha_reg": args.alpha_reg,
        "res_reg": args.res_reg,
        "alpha_init_bias": args.alpha_init_bias,
        "best_epoch": best_epoch,
        "best_val_mse": best_metric,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(out_dir, "sparse_moe_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nSparse MoE {args.run_name} test MSE/MAE: "
        f"{test_metrics['mse']:.6f} / {test_metrics['mae']:.6f} "
        f"(base {test_metrics['base_mse']:.6f}, best {test_metrics['best_mse']:.6f})"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train sparse residual latent MoE gate on frozen branches")
    parser.add_argument("--source_dir", type=str, default="./latent_outputs/official_multibranch_ETTh1_sl96_pl96_normK8")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/official_multibranch_ETTh1_sl96_pl96_normK8_sparse_moe")
    parser.add_argument("--top_m", type=int, default=2)
    parser.add_argument("--moe_tau", type=float, default=1.0)
    parser.add_argument("--alpha_init_bias", type=float, default=-3.0)
    parser.add_argument("--alpha_reg", type=float, default=0.0)
    parser.add_argument("--res_reg", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    top_name = "dense" if args.top_m <= 0 else f"top{args.top_m}"
    args.run_name = f"{top_name}_areg{args.alpha_reg:g}_rreg{args.res_reg:g}"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.source_dir, "args.json"), "r") as f:
        branch_args = SimpleNamespace(**json.load(f))
    branch_args.device = args.device
    args.num_branches = branch_args.num_branches
    args.enc_in = branch_args.enc_in

    summary = train_one(args, branch_args, device)
    with open(os.path.join(args.output_dir, f"summary_top{args.top_m}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
