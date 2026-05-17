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
from my_utils import model_dict


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


class FrozenDLinearMultiBranch(torch.nn.Module):
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
        z_base = z_base[:, -self.pred_len:, :]
        deltas = self.branch_delta(z_base)
        deltas = deltas.view(z_base.size(0), self.pred_len, self.num_branches, self.d_model)
        deltas = deltas.permute(0, 2, 1, 3).contiguous()
        z_branches = z_base[:, None] + deltas
        return z_base, z_branches

    def forward(self, z_x):
        z_base, z_branches = self.forward_with_base(z_x)
        context_hidden = self.context_proj(z_base.mean(dim=1))
        context_hidden = context_hidden[:, None, :].expand(-1, self.num_branches, -1)
        branch_hidden = self.branch_score_proj(z_branches.mean(dim=2))
        logits = self.scorer(torch.cat([context_hidden, branch_hidden], dim=-1)).squeeze(-1)
        return z_base, z_branches, logits


class LatentForecaster(torch.nn.Module):
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
        return z_pred[:, -self.pred_len:, :]


class CorrectionQualityScorer(torch.nn.Module):
    def __init__(self, context_dim, branch_dim, match_dim, hidden_dim, dropout):
        super().__init__()
        self.context_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(context_dim),
            torch.nn.Linear(context_dim, match_dim),
            torch.nn.GELU(),
        )
        self.branch_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(branch_dim),
            torch.nn.Linear(branch_dim, match_dim),
            torch.nn.GELU(),
        )
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(match_dim * 4),
            torch.nn.Linear(match_dim * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, z0, branch_embed):
        context = self.context_proj(z0)
        branch = self.branch_proj(branch_embed)
        context = context[:, None, :].expand_as(branch)
        pair = torch.cat([context, branch, context - branch, context * branch], dim=-1)
        return self.scorer(pair).squeeze(-1)


def load_branch_model(args, device):
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
    missing = [key for key in missing if not key.startswith("context_proj.") and not key.startswith("branch_score_proj.") and not key.startswith("scorer.")]
    if missing or unexpected:
        raise RuntimeError(f"Could not load frozen branch generator cleanly. missing={missing}, unexpected={unexpected}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, branch_args


def load_latent_forecaster(args, branch_args, device):
    if not args.latent_forecaster_checkpoint:
        return None
    checkpoint = torch.load(args.latent_forecaster_checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    lf_args = SimpleNamespace(**vars(branch_args))
    for key, value in ckpt_args.items():
        setattr(lf_args, key, value)
    model = LatentForecaster(lf_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def decode_branches(autoencoder, z_branches):
    bsz, branches, steps, dim = z_branches.shape
    y_branches = autoencoder.decode(z_branches.reshape(bsz * branches, steps, dim))
    return y_branches.reshape(bsz, branches, steps, -1)


def prepare_batch(args, autoencoder, branch_model, latent_forecaster, batch_x, batch_y, device):
    x = batch_x.float().to(device)
    y_true = batch_y[:, -args.pred_len:, :].float().to(device)
    with torch.no_grad():
        z_x = autoencoder.encode(x)
        z_base, z_branches = branch_model.forward_with_base(z_x)
        y_base = autoencoder.decode(z_base)
        y_branches = decode_branches(autoencoder, z_branches)
        branch_mse = ((y_branches - y_true[:, None]) ** 2).mean(dim=(2, 3))
        base_mse = ((y_base - y_true) ** 2).mean(dim=(1, 2))
        improvement = base_mse[:, None] - branch_mse
        labels = branch_mse.argmin(dim=1)
        y_latenttsf = None
        if latent_forecaster is not None:
            y_latenttsf = autoencoder.decode(latent_forecaster(z_x))
    if args.selector_input == "zx":
        z0 = z_x.mean(dim=1)
    elif args.selector_input == "zx_base":
        z0 = torch.cat([z_x.mean(dim=1), z_base.mean(dim=1)], dim=-1)
    else:
        raise ValueError(f"Unknown selector_input: {args.selector_input}")
    if args.branch_embedding == "branch":
        branch_embed = z_branches.mean(dim=2)
    elif args.branch_embedding == "delta":
        branch_embed = (z_branches - z_base[:, None]).mean(dim=2)
    else:
        raise ValueError(f"Unknown branch_embedding: {args.branch_embedding}")
    return (
        z0.detach(),
        branch_embed.detach(),
        labels.detach(),
        branch_mse.detach(),
        improvement.detach(),
        y_branches.detach(),
        y_true.detach(),
        y_base.detach(),
        y_latenttsf,
    )


def quality_loss(args, scores, improvement):
    target = improvement.detach() * args.label_scale
    reg = F.mse_loss(scores, target)
    if args.quality_loss == "regression":
        return reg

    diff_score = scores[:, :, None] - scores[:, None, :]
    diff_target = target[:, :, None] - target[:, None, :]
    mask = diff_target.abs() > args.rank_eps
    if not mask.any():
        rank = scores.new_zeros(())
    else:
        rank = F.softplus(-diff_score[mask] * diff_target[mask].sign()).mean()
    if args.quality_loss == "ranking":
        return rank
    return reg + args.rank_weight * rank


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def topk_coverage_from_counts(counts, k=5):
    counts = np.asarray(counts, dtype=np.float64)
    return float(np.sort(counts)[::-1][:k].sum() / max(counts.sum(), 1.0))


def kl_from_counts(p_counts, q_counts, eps=1e-12):
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    return float((p * (np.log(p) - np.log(q))).sum())


def collect_winner_stats(args, autoencoder, branch_model, latent_forecaster, loader, device):
    counts = np.zeros(args.num_branches, dtype=np.int64)
    total = 0
    oracle_sum = 0.0
    base_sum = 0.0
    latenttsf_sum = 0.0
    has_latenttsf = latent_forecaster is not None
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch = prepare_batch(args, autoencoder, branch_model, latent_forecaster, batch_x, batch_y, device)
            _, _, labels, branch_mse, _, _, y_true, y_base, y_latenttsf = batch
            bsz = labels.size(0)
            total += bsz
            counts += np.bincount(labels.cpu().numpy(), minlength=args.num_branches)
            oracle_sum += branch_mse.min(dim=-1).values.sum().item()
            base_sum += F.mse_loss(y_base, y_true, reduction="sum").item() / (y_true.size(1) * y_true.size(2))
            if has_latenttsf:
                latenttsf_sum += F.mse_loss(y_latenttsf, y_true, reduction="sum").item() / (
                    y_true.size(1) * y_true.size(2)
                )
    stats = {
        "counts": counts.tolist(),
        "ratios": (counts / max(total, 1)).tolist(),
        "entropy": entropy_from_counts(counts),
        "top5_coverage": topk_coverage_from_counts(counts, min(5, args.num_branches)),
        "oracle_best_mse": oracle_sum / max(total, 1),
        "dlinear_branch_baseline_mse": base_sum / max(total, 1),
    }
    if has_latenttsf:
        stats["latenttsf_baseline_mse"] = latenttsf_sum / max(total, 1)
    return stats


def evaluate(args, scorer, autoencoder, branch_model, latent_forecaster, loader, device, seed):
    scorer.eval()
    rng = np.random.default_rng(seed)
    sums = {
        "ce": 0.0,
        "winner_at_1": 0.0,
        "winner_at_3": 0.0,
        "oracle_best_mse": 0.0,
        "selected_branch_mse": 0.0,
        "random_branch_mse": 0.0,
        "expected_branch_mse": 0.0,
        "weighted_correction_mse": 0.0,
        "dlinear_branch_baseline_mse": 0.0,
        "latenttsf_baseline_mse": 0.0,
        "mean_positive_improvement_rate": 0.0,
        "entropy": 0.0,
        "max_prob": 0.0,
    }
    count = 0
    has_latenttsf = latent_forecaster is not None
    winner_counts = np.zeros(args.num_branches, dtype=np.int64)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            z0, branch_embed, labels, branch_mse, improvement, y_branches, y_true, y_base, y_latenttsf = prepare_batch(
                args, autoencoder, branch_model, latent_forecaster, batch_x, batch_y, device
            )
            scores = scorer(z0, branch_embed)
            weights = torch.softmax(scores / args.score_tau, dim=-1)
            selected = scores.argmax(dim=-1)
            top3 = scores.topk(min(3, scores.size(-1)), dim=-1).indices
            random_idx = torch.from_numpy(rng.integers(0, scores.size(-1), size=z0.size(0))).long().to(device)
            delta_y = y_branches - y_base[:, None]
            y_weighted = y_base + args.correction_alpha * torch.einsum("bk,bktc->btc", weights, delta_y)

            bsz = z0.size(0)
            count += bsz
            sums["ce"] += F.cross_entropy(scores, labels, reduction="sum").item()
            winner_counts += np.bincount(labels.cpu().numpy(), minlength=args.num_branches)
            sums["winner_at_1"] += (selected == labels).float().sum().item()
            sums["winner_at_3"] += (top3 == labels[:, None]).any(dim=-1).float().sum().item()
            sums["oracle_best_mse"] += branch_mse.min(dim=-1).values.sum().item()
            sums["selected_branch_mse"] += branch_mse.gather(1, selected[:, None]).sum().item()
            sums["random_branch_mse"] += branch_mse.gather(1, random_idx[:, None]).sum().item()
            sums["expected_branch_mse"] += (weights * branch_mse).sum(dim=-1).sum().item()
            sums["weighted_correction_mse"] += F.mse_loss(y_weighted, y_true, reduction="sum").item() / (
                y_true.size(1) * y_true.size(2)
            )
            sums["dlinear_branch_baseline_mse"] += F.mse_loss(y_base, y_true, reduction="sum").item() / (
                y_true.size(1) * y_true.size(2)
            )
            if has_latenttsf:
                sums["latenttsf_baseline_mse"] += F.mse_loss(y_latenttsf, y_true, reduction="sum").item() / (
                    y_true.size(1) * y_true.size(2)
                )
            sums["mean_positive_improvement_rate"] += (improvement > 0).float().mean(dim=-1).sum().item()
            sums["entropy"] += (-(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)).sum().item()
            sums["max_prob"] += weights.max(dim=-1).values.sum().item()

    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["winner_counts"] = winner_counts.tolist()
    result["winner_entropy"] = entropy_from_counts(winner_counts)
    result["winner_top5_coverage"] = topk_coverage_from_counts(winner_counts, min(5, args.num_branches))
    if not has_latenttsf:
        result.pop("latenttsf_baseline_mse")
    return result


def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    autoencoder = load_autoencoder(args, device)
    branch_model, branch_args = load_branch_model(args, device)
    latent_forecaster = load_latent_forecaster(args, branch_args, device)

    input_dim = args.d_model if args.selector_input == "zx" else args.d_model * 2
    scorer = CorrectionQualityScorer(
        context_dim=input_dim,
        branch_dim=args.d_model,
        match_dim=args.match_dim,
        hidden_dim=args.selector_hidden_dim,
        dropout=args.selector_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    train_winner_stats = collect_winner_stats(args, autoencoder, branch_model, latent_forecaster, train_loader, device)
    val_winner_stats = collect_winner_stats(args, autoencoder, branch_model, latent_forecaster, val_loader, device)
    test_winner_stats = collect_winner_stats(args, autoencoder, branch_model, latent_forecaster, test_loader, device)
    winner_distribution = {
        "train": train_winner_stats,
        "val": val_winner_stats,
        "test": test_winner_stats,
        "kl": {
            "train_to_val": kl_from_counts(train_winner_stats["counts"], val_winner_stats["counts"]),
            "train_to_test": kl_from_counts(train_winner_stats["counts"], test_winner_stats["counts"]),
            "val_to_test": kl_from_counts(val_winner_stats["counts"], test_winner_stats["counts"]),
        },
    }

    print(
        "Frozen DLinear branch selector | "
        f"AE frozen=True branch frozen=True K={branch_args.num_branches} "
        f"selector_input={args.selector_input} branch_embedding={args.branch_embedding} "
        f"quality_loss={args.quality_loss} alpha={args.correction_alpha} tau={args.score_tau}",
        flush=True,
    )
    print("Winner distribution diagnostics:")
    for name, stats in [("train", train_winner_stats), ("val", val_winner_stats), ("test", test_winner_stats)]:
        print(
            f"  {name}: counts={stats['counts']} entropy={stats['entropy']:.4f} "
            f"top5={stats['top5_coverage']:.4f} oracle={stats['oracle_best_mse']:.6f} "
            f"base={stats['dlinear_branch_baseline_mse']:.6f}",
            flush=True,
        )
    print(f"  KL train->val={winner_distribution['kl']['train_to_val']:.4f} "
          f"train->test={winner_distribution['kl']['train_to_test']:.4f} "
          f"val->test={winner_distribution['kl']['val_to_test']:.4f}", flush=True)

    best_state = None
    best_epoch = 0
    best_metric = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        scorer.train()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            z0, branch_embed, labels, branch_mse, improvement, y_branches, y_true, y_base, y_latenttsf = prepare_batch(
                args, autoencoder, branch_model, latent_forecaster, batch_x, batch_y, device
            )
            scores = scorer(z0, branch_embed)
            loss = quality_loss(args, scores, improvement)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * z0.size(0)
            total_count += z0.size(0)

        train_metrics = evaluate(args, scorer, autoencoder, branch_model, latent_forecaster, train_loader, device, args.seed + epoch)
        val_metrics = evaluate(args, scorer, autoencoder, branch_model, latent_forecaster, val_loader, device, args.seed + epoch + 1000)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch:03d} | "
            f"train loss {total_loss / max(total_count, 1):.4f} "
            f"w@1 {train_metrics['winner_at_1']:.4f} sel/rand/oracle "
            f"{train_metrics['selected_branch_mse']:.6f}/{train_metrics['random_branch_mse']:.6f}/"
            f"{train_metrics['oracle_best_mse']:.6f} base {train_metrics['dlinear_branch_baseline_mse']:.6f} | "
            f"val w@1 {val_metrics['winner_at_1']:.4f} w@3 {val_metrics['winner_at_3']:.4f} "
            f"sel/rand/oracle {val_metrics['selected_branch_mse']:.6f}/"
            f"{val_metrics['random_branch_mse']:.6f}/{val_metrics['oracle_best_mse']:.6f} "
            f"base {val_metrics['dlinear_branch_baseline_mse']:.6f}",
            flush=True,
        )

        metric = val_metrics["selected_branch_mse"]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()}
            torch.save(
                {
                    "scorer_state_dict": best_state,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "best_frozen_dlinear_branch_selector.pt"),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        scorer.load_state_dict(best_state)

    val_metrics = evaluate(args, scorer, autoencoder, branch_model, latent_forecaster, val_loader, device, args.seed + 2000)
    test_metrics = evaluate(args, scorer, autoencoder, branch_model, latent_forecaster, test_loader, device, args.seed + 3000)
    summary = {
        "best_epoch": best_epoch,
        "branch_checkpoint": args.branch_checkpoint,
        "latent_forecaster_checkpoint": args.latent_forecaster_checkpoint,
        "winner_distribution": winner_distribution,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(args.output_dir, "frozen_dlinear_branch_selector_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFrozen DLinear Future-Aligned Branch Selector [test]")
    for key in [
        "winner_at_1",
        "winner_at_3",
        "oracle_best_mse",
        "selected_branch_mse",
        "random_branch_mse",
        "expected_branch_mse",
        "weighted_correction_mse",
        "dlinear_branch_baseline_mse",
        "latenttsf_baseline_mse",
    ]:
        if key in test_metrics:
            print(f"  {key}: {test_metrics[key]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Future-aligned selector on frozen AE + frozen DLinear branch model")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/frozen_dlinear_branch_selector")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--branch_checkpoint", type=str, required=True)
    parser.add_argument("--latent_forecaster_checkpoint", type=str, default="")

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

    parser.add_argument("--selector_input", type=str, default="zx", choices=["zx", "zx_base"])
    parser.add_argument("--selector_hidden_dim", type=int, default=64)
    parser.add_argument("--selector_dropout", type=float, default=0.1)
    parser.add_argument("--match_dim", type=int, default=64)
    parser.add_argument("--branch_embedding", type=str, default="branch", choices=["branch", "delta"])
    parser.add_argument("--quality_loss", type=str, default="hybrid", choices=["regression", "ranking", "hybrid"])
    parser.add_argument("--rank_weight", type=float, default=0.1)
    parser.add_argument("--rank_eps", type=float, default=1e-4)
    parser.add_argument("--label_scale", type=float, default=1.0)
    parser.add_argument("--score_tau", type=float, default=0.2)
    parser.add_argument("--correction_alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
