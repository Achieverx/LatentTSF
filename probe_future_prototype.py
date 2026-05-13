import argparse
import json
import os
import random

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MeanPoolProbe(nn.Module):
    def __init__(self, d_model, num_classes, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_x):
        pooled = z_x.mean(dim=1)
        return self.classifier(pooled)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def topk_accuracy(logits, labels, k):
    topk = logits.topk(k, dim=1).indices
    return (topk == labels[:, None]).any(dim=1).float().mean().item()


def per_class_accuracy(cm):
    denom = cm.sum(axis=1)
    return np.divide(
        np.diag(cm),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom != 0,
    )


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Probe confusion matrix")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    max_value = cm.max() if cm.size else 0
    threshold = max_value / 2 if max_value else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    pred = logits_all.argmax(dim=1).numpy()
    labels_np = labels_all.numpy()
    cm = confusion_matrix(labels_np, pred, labels=np.arange(num_classes))

    return {
        "loss": total_loss / max(total_count, 1),
        "top1_acc": accuracy(logits_all, labels_all),
        "top3_acc": topk_accuracy(logits_all, labels_all, min(3, num_classes)),
        "balanced_acc": balanced_accuracy_score(labels_np, pred),
        "confusion_matrix": cm,
        "per_class_acc": per_class_accuracy(cm),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe future prototype predictability")
    parser.add_argument(
        "--latent_dir",
        type=str,
        default="./latent_outputs/ETTh1_sl96_pl96_dm32_dff64_MLP",
    )
    parser.add_argument("--zx_file", type=str, default="train_zx.npy")
    parser.add_argument("--label_file", type=str, default="zy_kmeanspp_labels_K8.npy")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = args.out_dir
    if out_dir is None:
        suffix = "weighted" if args.class_weight else "unweighted"
        out_dir = os.path.join(args.latent_dir, f"probe_k8_{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    z_x = np.load(os.path.join(args.latent_dir, args.zx_file)).astype(np.float32)
    labels = np.load(os.path.join(args.latent_dir, args.label_file)).astype(np.int64)

    if z_x.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatched sample count: z_x={z_x.shape}, labels={labels.shape}")

    n_samples = z_x.shape[0]
    split = int(n_samples * args.train_ratio)
    train_x, val_x = z_x[:split], z_x[split:]
    train_y, val_y = labels[:split], labels[split:]

    train_counts = np.bincount(train_y, minlength=args.num_classes)
    val_counts = np.bincount(val_y, minlength=args.num_classes)
    total_counts = np.bincount(labels, minlength=args.num_classes)
    majority_baseline = val_counts.max() / max(val_counts.sum(), 1)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MeanPoolProbe(
        d_model=z_x.shape[-1],
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if args.class_weight:
        weights = train_counts.sum() / (args.num_classes * np.maximum(train_counts, 1))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        weights = None
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_metric = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = os.path.join(out_dir, "best_probe.pt")

    print(f"Loaded z_x: {z_x.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Train/val split: {split}/{n_samples - split}")
    print(f"Total counts: {total_counts.tolist()}")
    print(f"Train counts: {train_counts.tolist()}")
    print(f"Val counts: {val_counts.tolist()}")
    print(f"Random top-1 baseline: {1 / args.num_classes:.4f}")
    print(f"Random top-3 baseline: {min(3, args.num_classes) / args.num_classes:.4f}")
    print(f"Val majority baseline: {majority_baseline:.4f}")
    print(f"Class weighting: {args.class_weight}")
    if weights is not None:
        print(f"Class weights: {[round(float(w), 4) for w in weights]}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        train_eval = evaluate(model, train_loader, criterion, device, args.num_classes)
        val_eval = evaluate(model, val_loader, criterion, device, args.num_classes)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_top1_acc": train_eval["top1_acc"],
            "train_balanced_acc": train_eval["balanced_acc"],
            "val_loss": val_eval["loss"],
            "val_top1_acc": val_eval["top1_acc"],
            "val_top3_acc": val_eval["top3_acc"],
            "val_balanced_acc": val_eval["balanced_acc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.4f} top1 {row['train_top1_acc']:.4f} bal {row['train_balanced_acc']:.4f} | "
            f"val loss {row['val_loss']:.4f} top1 {row['val_top1_acc']:.4f} "
            f"top3 {row['val_top3_acc']:.4f} bal {row['val_balanced_acc']:.4f}",
            flush=True,
        )

        current_metric = val_eval["balanced_acc"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_balanced_acc": current_metric,
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
    train_eval = evaluate(model, train_loader, criterion, device, args.num_classes)
    val_eval = evaluate(model, val_loader, criterion, device, args.num_classes)

    cm = val_eval["confusion_matrix"]
    cm_path = os.path.join(out_dir, "confusion_matrix.npy")
    cm_png_path = os.path.join(out_dir, "confusion_matrix.png")
    np.save(cm_path, cm)
    plot_confusion_matrix(cm, cm_png_path)

    metrics = {
        "best_epoch": best_epoch,
        "class_weight": args.class_weight,
        "random_top1_baseline": 1 / args.num_classes,
        "random_top3_baseline": min(3, args.num_classes) / args.num_classes,
        "val_majority_baseline": majority_baseline,
        "total_counts": total_counts.tolist(),
        "train_counts": train_counts.tolist(),
        "val_counts": val_counts.tolist(),
        "train": {
            "loss": train_eval["loss"],
            "top1_acc": train_eval["top1_acc"],
            "top3_acc": train_eval["top3_acc"],
            "balanced_acc": train_eval["balanced_acc"],
            "per_class_acc": train_eval["per_class_acc"].tolist(),
        },
        "val": {
            "loss": val_eval["loss"],
            "top1_acc": val_eval["top1_acc"],
            "top3_acc": val_eval["top3_acc"],
            "balanced_acc": val_eval["balanced_acc"],
            "per_class_acc": val_eval["per_class_acc"].tolist(),
        },
        "history": history,
        "confusion_matrix_path": cm_path,
        "confusion_matrix_png_path": cm_png_path,
        "best_probe_path": best_path,
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nBest probe metrics")
    print(f"  best_epoch: {best_epoch}")
    print(f"  val top-1: {metrics['val']['top1_acc']:.4f}")
    print(f"  val top-3: {metrics['val']['top3_acc']:.4f}")
    print(f"  val balanced acc: {metrics['val']['balanced_acc']:.4f}")
    print(f"  val majority baseline: {majority_baseline:.4f}")
    print(f"  val per-class acc: {[round(x, 4) for x in metrics['val']['per_class_acc']]}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {cm_png_path}")


if __name__ == "__main__":
    main()
