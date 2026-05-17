import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from train_frozen_dlinear_branch_selector import FrozenDLinearMultiBranch


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def official_loader(args, flag, shuffle=False):
    data_set, _ = data_provider(args, flag)
    return DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
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


def load_branch_bank(args, device):
    checkpoint = torch.load(args.branch_checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    branch_args = SimpleNamespace(**ckpt_args)
    for key in [
        "model",
        "pred_len",
        "d_model",
        "d_ff",
        "moving_avg",
        "individual",
        "num_branches",
        "hidden_dim",
        "dropout",
    ]:
        if not hasattr(branch_args, key):
            setattr(branch_args, key, getattr(args, key))

    model = FrozenDLinearMultiBranch(branch_args).to(device)
    state = checkpoint["model_state_dict"]
    branch_state = {
        key: value
        for key, value in state.items()
        if key.startswith("backbone.") or key.startswith("branch_delta.")
    }
    missing, unexpected = model.load_state_dict(branch_state, strict=False)
    missing = [
        key
        for key in missing
        if not key.startswith("context_proj.")
        and not key.startswith("branch_score_proj.")
        and not key.startswith("scorer.")
    ]
    if missing or unexpected:
        raise RuntimeError(f"Could not load branch bank. missing={missing}, unexpected={unexpected}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, branch_args


class BranchProbe(torch.nn.Module):
    def __init__(self, d_model, hidden_dim, num_branches, dropout):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.mode_head = torch.nn.Linear(hidden_dim, num_branches)
        self.quality_head = torch.nn.Linear(hidden_dim, num_branches)

    def forward(self, z_x):
        # Stopgrad is enforced by callers; keep it explicit here too.
        h = self.trunk(z_x.detach().mean(dim=1))
        return self.mode_head(h), self.quality_head(h)


def decode_branches(autoencoder, z_branches):
    bsz, branches, steps, dim = z_branches.shape
    y_branches = autoencoder.decode(z_branches.reshape(bsz * branches, steps, dim))
    return y_branches.reshape(bsz, branches, steps, -1)


def branch_targets(args, autoencoder, branch_bank, z_x, y_true):
    with torch.no_grad():
        z_base, z_branches = branch_bank.forward_with_base(z_x)
        y_base = autoencoder.decode(z_base)
        y_branches = decode_branches(autoencoder, z_branches)
        errors = ((y_branches - y_true[:, None]) ** 2).mean(dim=(2, 3))
        base_error = ((y_base - y_true) ** 2).mean(dim=(1, 2))
        improvement = base_error[:, None] - errors
        q = torch.softmax(-errors / args.branch_tau, dim=-1)
        labels = errors.argmin(dim=-1)
        diversity = pairwise_diversity(y_branches)
    return q, labels, errors, improvement, base_error, diversity, y_base, y_branches


def pairwise_diversity(y_branches):
    if y_branches.size(1) < 2:
        return y_branches.new_zeros(())
    flat = y_branches.flatten(start_dim=2)
    values = []
    for i in range(y_branches.size(1)):
        for j in range(i + 1, y_branches.size(1)):
            values.append(((flat[:, i] - flat[:, j]) ** 2).mean(dim=1))
    return torch.stack(values, dim=1).mean()


def encode_batch(args, autoencoder, batch_x, batch_y, device):
    x = batch_x.float().to(device)
    y = batch_y[:, -args.pred_len:, :].float().to(device)
    with torch.no_grad():
        z_x = autoencoder.encode(x)
    return y, z_x


def quality_corr(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.pow(2).mean().sqrt() * target.pow(2).mean().sqrt()
    if denom.item() <= 1e-12:
        return pred.new_zeros(())
    return (pred * target).mean() / denom


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def topk_coverage(counts, k=5):
    counts = np.asarray(counts, dtype=np.float64)
    return float(np.sort(counts)[::-1][:k].sum() / max(counts.sum(), 1.0))


def kl_counts(p_counts, q_counts, eps=1e-12):
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    return float((p * (np.log(p) - np.log(q))).sum())


def parse_alpha_sweep(alpha_sweep):
    if isinstance(alpha_sweep, (list, tuple)):
        return [float(value) for value in alpha_sweep]
    return [float(value.strip()) for value in alpha_sweep.split(",") if value.strip()]


def correction_weights(args, quality):
    if args.correction_weight == "softmax":
        return torch.softmax(quality / args.correction_tau, dim=-1)
    positive = torch.relu(quality)
    return positive / (positive.sum(dim=-1, keepdim=True) + args.safe_eps)


def compute_probe_loss(args, probe, autoencoder, branch_bank, batch_x, batch_y, device):
    y_true, z_x = encode_batch(args, autoencoder, batch_x, batch_y, device)
    q, labels, errors, improvement, base_error, diversity, _, _ = branch_targets(
        args, autoencoder, branch_bank, z_x, y_true
    )
    logits, quality = probe(z_x.detach())
    mode_loss = F.kl_div(F.log_softmax(logits, dim=-1), q, reduction="batchmean")
    quality_loss = F.mse_loss(quality, improvement.detach() * args.quality_scale)
    total = mode_loss + args.lambda_quality * quality_loss
    top3 = logits.topk(min(3, logits.size(-1)), dim=-1).indices
    pred = logits.argmax(dim=-1)
    return total, {
        "loss": total.detach(),
        "mode_kl": mode_loss.detach(),
        "quality_mse": quality_loss.detach(),
        "quality_corr": quality_corr(quality.detach(), improvement.detach()),
        "winner_at_1": (pred == labels).float().mean().detach(),
        "winner_at_3": (top3 == labels[:, None]).any(dim=-1).float().mean().detach(),
        "oracle_best_mse": errors.min(dim=-1).values.mean().detach(),
        "base_mse": base_error.mean().detach(),
        "branch_diversity": diversity.detach(),
        "labels": labels.detach(),
    }


def evaluate(args, probe, autoencoder, branch_bank, loader, device):
    probe.eval()
    alphas = parse_alpha_sweep(args.alpha_sweep)
    sums = {
        "loss": 0.0,
        "mode_kl": 0.0,
        "quality_mse": 0.0,
        "quality_corr": 0.0,
        "winner_at_1": 0.0,
        "winner_at_3": 0.0,
        "oracle_best_mse": 0.0,
        "base_mse": 0.0,
        "branch_diversity": 0.0,
        "mean_predicted_positive_branches": 0.0,
    }
    correction_sums = {
        str(alpha): {"mse": 0.0, "mae": 0.0, "harmful": 0.0, "gain": 0.0}
        for alpha in alphas
    }
    count = 0
    winner_counts = np.zeros(args.num_branches, dtype=np.int64)
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            y_true, z_x = encode_batch(args, autoencoder, batch_x, batch_y, device)
            q, labels, errors, improvement, base_error, diversity, y_base, y_branches = branch_targets(
                args, autoencoder, branch_bank, z_x, y_true
            )
            logits, quality = probe(z_x.detach())
            mode_loss = F.kl_div(F.log_softmax(logits, dim=-1), q, reduction="batchmean")
            quality_loss = F.mse_loss(quality, improvement.detach() * args.quality_scale)
            total = mode_loss + args.lambda_quality * quality_loss
            top3 = logits.topk(min(3, logits.size(-1)), dim=-1).indices
            pred = logits.argmax(dim=-1)
            metrics = {
                "loss": total.detach(),
                "mode_kl": mode_loss.detach(),
                "quality_mse": quality_loss.detach(),
                "quality_corr": quality_corr(quality.detach(), improvement.detach()),
                "winner_at_1": (pred == labels).float().mean().detach(),
                "winner_at_3": (top3 == labels[:, None]).any(dim=-1).float().mean().detach(),
                "oracle_best_mse": errors.min(dim=-1).values.mean().detach(),
                "base_mse": base_error.mean().detach(),
                "branch_diversity": diversity.detach(),
                "mean_predicted_positive_branches": (quality > 0).float().sum(dim=-1).mean().detach(),
                "labels": labels.detach(),
            }
            bsz = batch_x.size(0)
            count += bsz
            for key in sums:
                sums[key] += metrics[key].item() * bsz
            winner_counts += np.bincount(metrics["labels"].cpu().numpy(), minlength=args.num_branches)

            weights = correction_weights(args, quality)
            deltas = y_branches - y_base[:, None]
            safe_delta = torch.einsum("bk,bktc->btc", weights, deltas)
            for alpha in alphas:
                key = str(alpha)
                y_safe = y_base + alpha * safe_delta
                mse_per_sample = ((y_safe - y_true) ** 2).mean(dim=(1, 2))
                mae_per_sample = (y_safe - y_true).abs().mean(dim=(1, 2))
                correction_sums[key]["mse"] += mse_per_sample.sum().item()
                correction_sums[key]["mae"] += mae_per_sample.sum().item()
                correction_sums[key]["harmful"] += (mse_per_sample > base_error).float().sum().item()
                correction_sums[key]["gain"] += (base_error - mse_per_sample).sum().item()
    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["baseline_mse"] = result["base_mse"]
    result["winner_counts"] = winner_counts.tolist()
    result["winner_entropy"] = entropy_from_counts(winner_counts)
    result["winner_top5_coverage"] = topk_coverage(winner_counts, min(5, args.num_branches))
    result["safe_correction"] = {
        key: {
            "mse": value["mse"] / max(count, 1),
            "mae": value["mae"] / max(count, 1),
            "harmful_ratio": value["harmful"] / max(count, 1),
            "mean_gain_vs_baseline": value["gain"] / max(count, 1),
        }
        for key, value in correction_sums.items()
    }
    return result


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    branch_bank, branch_args = load_branch_bank(args, device)
    args.num_branches = branch_args.num_branches
    probe = BranchProbe(args.d_model, args.probe_hidden_dim, branch_args.num_branches, args.dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    print(
        f"Branch probe only | K={branch_args.num_branches} tau={args.branch_tau} "
        f"lambda_quality={args.lambda_quality}",
        flush=True,
    )
    print("Only H is trainable. AE/branch bank/forecast model are not updated.", flush=True)

    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        train_count = 0
        train_loss = 0.0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            loss, metrics = compute_probe_loss(args, probe, autoencoder, branch_bank, batch_x, batch_y, device)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            train_count += batch_x.size(0)

        train_metrics = evaluate(args, probe, autoencoder, branch_bank, train_loader, device)
        val_metrics = evaluate(args, probe, autoencoder, branch_bank, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss / max(train_count, 1):.4f} "
            f"w@1 {train_metrics['winner_at_1']:.3f} w@3 {train_metrics['winner_at_3']:.3f} "
            f"corr {train_metrics['quality_corr']:.3f} | "
            f"val KL {val_metrics['mode_kl']:.4f} w@1 {val_metrics['winner_at_1']:.3f} "
            f"w@3 {val_metrics['winner_at_3']:.3f} corr {val_metrics['quality_corr']:.3f}",
            flush=True,
        )
        metric = val_metrics[args.early_stop_metric]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            torch.save(
                {
                    "probe_state_dict": best_state,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "best_branch_probe.pt"),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    train_metrics = evaluate(args, probe, autoencoder, branch_bank, train_loader, device)
    val_metrics = evaluate(args, probe, autoencoder, branch_bank, val_loader, device)
    test_metrics = evaluate(args, probe, autoencoder, branch_bank, test_loader, device)
    summary = {
        "best_epoch": best_epoch,
        "branch_checkpoint": args.branch_checkpoint,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "drift": {
            "train_to_val_kl": kl_counts(train_metrics["winner_counts"], val_metrics["winner_counts"]),
            "train_to_test_kl": kl_counts(train_metrics["winner_counts"], test_metrics["winner_counts"]),
            "val_to_test_kl": kl_counts(val_metrics["winner_counts"], test_metrics["winner_counts"]),
        },
    }
    with open(os.path.join(args.output_dir, "branch_probe_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBranch probe [test]")
    print(f"  baseline MSE:         {test_metrics['baseline_mse']:.6f}")
    print(f"  oracle best-of-K MSE: {test_metrics['oracle_best_mse']:.6f}")
    print(f"  branch diversity:     {test_metrics['branch_diversity']:.6f}")
    print(f"  winner@1/@3:          {test_metrics['winner_at_1']:.4f} / {test_metrics['winner_at_3']:.4f}")
    print(f"  KL(q||p):             {test_metrics['mode_kl']:.6f}")
    print(f"  quality corr:         {test_metrics['quality_corr']:.4f}")
    print(f"  mean positive preds:  {test_metrics['mean_predicted_positive_branches']:.4f}")
    print(f"  safe correction ({args.correction_weight})")
    for alpha, metrics in test_metrics["safe_correction"].items():
        print(
            f"    alpha={float(alpha):.3f}: MSE {metrics['mse']:.6f} / MAE {metrics['mae']:.6f} "
            f"harm {metrics['harmful_ratio']:.4f} gain {metrics['mean_gain_vs_baseline']:.6f}"
        )
    print(f"  drift train->test KL: {summary['drift']['train_to_test_kl']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Probe whether frozen latent contains branch information")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/branch_probe")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--branch_checkpoint", type=str, required=True)
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
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_branches", type=int, default=8)
    parser.add_argument("--probe_hidden_dim", type=int, default=128)
    parser.add_argument("--branch_tau", type=float, default=0.1)
    parser.add_argument("--lambda_quality", type=float, default=1.0)
    parser.add_argument("--quality_scale", type=float, default=1.0)
    parser.add_argument("--alpha_sweep", type=str, default="0.01,0.03,0.05,0.1")
    parser.add_argument("--correction_weight", type=str, default="relu", choices=["relu", "softmax"])
    parser.add_argument("--correction_tau", type=float, default=0.2)
    parser.add_argument("--safe_eps", type=float, default=1e-8)
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
        default="mode_kl",
        choices=["loss", "mode_kl", "quality_mse"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
