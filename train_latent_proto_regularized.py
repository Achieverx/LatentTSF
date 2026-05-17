import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd  # preload before torch to avoid a pyarrow access violation on Windows
from datasets import load_dataset  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
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


def patchify(seq, patch_len):
    if seq.shape[1] % patch_len != 0:
        raise ValueError(f"length={seq.shape[1]} must be divisible by patch_len={patch_len}.")
    return seq.reshape(seq.shape[0], seq.shape[1] // patch_len, patch_len * seq.shape[2])


def normalize_patches_np(patches, eps):
    mean = patches.mean(axis=-1, keepdims=True)
    std = np.maximum(patches.std(axis=-1, keepdims=True), eps)
    return (patches - mean) / std


def normalize_patches_torch(patches, eps):
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (patches - mean) / std


def collect_future_latent_patches(args, autoencoder, device):
    loader = official_loader(args, "train", shuffle=False)
    patches = []
    total_seen = 0
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            y = batch_y[:, -args.pred_len:, :].float().to(device)
            z_y = autoencoder.encode(y)
            batch_patches = patchify(z_y.cpu().numpy().astype(np.float32), args.proto_patch_len)
            batch_patches = batch_patches.reshape(-1, batch_patches.shape[-1])

            if args.kmeans_max_samples > 0 and total_seen + batch_patches.shape[0] > args.kmeans_max_samples:
                remaining = args.kmeans_max_samples - total_seen
                if remaining <= 0:
                    break
                idx = rng.choice(batch_patches.shape[0], size=remaining, replace=False)
                batch_patches = batch_patches[idx]

            patches.append(batch_patches)
            total_seen += batch_patches.shape[0]

    return np.concatenate(patches, axis=0)


def fit_codebook(args, autoencoder, device):
    patches = collect_future_latent_patches(args, autoencoder, device)
    if args.normalize_proto_patches:
        patches = normalize_patches_np(patches, args.norm_eps)
    kmeans = MiniBatchKMeans(
        n_clusters=args.codebook_size,
        init="k-means++",
        batch_size=args.kmeans_batch_size,
        max_iter=args.kmeans_max_iter,
        n_init=args.kmeans_n_init,
        random_state=args.seed,
        verbose=0,
    )
    kmeans.fit(patches)
    return kmeans.cluster_centers_.astype(np.float32)


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


def delta_loss(z_pred, z_y):
    if z_pred.size(1) <= 1:
        return z_pred.new_tensor(0.0)
    return F.mse_loss(z_pred[:, 1:, :] - z_pred[:, :-1, :], z_y[:, 1:, :] - z_y[:, :-1, :])


def cosine_similarity_loss(z_pred, z_y):
    pred_flat = z_pred.reshape(-1, z_pred.shape[-1])
    target_flat = z_y.reshape(-1, z_y.shape[-1])
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()


def proto_regularization(z_pred, codebook, patch_len, tau, normalize_patches, norm_eps, distance):
    patches = patchify(z_pred, patch_len)
    flat = patches.reshape(-1, patches.shape[-1])
    if normalize_patches:
        flat = normalize_patches_torch(flat, norm_eps)

    if distance == "sq_l2_mean":
        diff = flat[:, None, :] - codebook[None, :, :]
        dist = diff.pow(2).mean(dim=-1).reshape(patches.shape[0], patches.shape[1], codebook.shape[0])
    elif distance == "l2_mean":
        dist = torch.cdist(flat, codebook, p=2).div(flat.shape[-1] ** 0.5).reshape(
            patches.shape[0], patches.shape[1], codebook.shape[0]
        )
    else:
        dist = torch.cdist(flat, codebook, p=2).reshape(
            patches.shape[0], patches.shape[1], codebook.shape[0]
        )

    if tau <= 0:
        return dist.min(dim=-1).values.mean()

    weights = torch.softmax(-dist / tau, dim=-1)
    return (weights * dist).sum(dim=-1).mean()


def encode_batch(args, autoencoder, batch_x, batch_y, device):
    x = batch_x.float().to(device)
    y = batch_y[:, -args.pred_len:, :].float().to(device)
    with torch.no_grad():
        z_x = autoencoder.encode(x)
        z_y = autoencoder.encode(y)
    return y, z_x, z_y


def compute_losses(args, autoencoder, model, codebook, batch_x, batch_y, device):
    y, z_x, z_y = encode_batch(args, autoencoder, batch_x, batch_y, device)
    z_pred = model(z_x)
    y_pred = autoencoder.decode(z_pred)

    loss_pred = F.mse_loss(y_pred, y)
    loss_latent = F.mse_loss(z_pred, z_y)
    loss_cosine = cosine_similarity_loss(z_pred, z_y)
    loss_delta = delta_loss(z_pred, z_y)
    loss_proto = proto_regularization(
        z_pred,
        codebook,
        args.proto_patch_len,
        args.proto_tau,
        args.normalize_proto_patches,
        args.norm_eps,
        args.proto_distance,
    )
    obs_mae = (y_pred - y).abs().mean()
    if args.loss_mode == "official_latent" and args.reconstruction_weight != 0:
        # With a frozen AE this term is constant w.r.t. the forecaster, but keep
        # it for official-loss accounting when explicitly requested.
        with torch.no_grad():
            y_recon = autoencoder.decode(z_y)
            loss_recon = F.l1_loss(y_recon, y)
    else:
        loss_recon = z_pred.new_zeros(())

    if args.loss_mode == "official_latent":
        total = (
            args.mse_weight * loss_latent
            + args.cosine_weight * loss_cosine
            + args.perceptual_weight * loss_pred
            + args.reconstruction_weight * loss_recon
            + args.lambda_proto * loss_proto
        )
    else:
        total = (
            loss_pred
            + args.lambda_latent * loss_latent
            + args.lambda_delta * loss_delta
            + args.lambda_proto * loss_proto
        )

    return total, {
        "total": total,
        "pred_mse": loss_pred,
        "pred_mae": obs_mae,
        "latent_mse": loss_latent,
        "cosine_loss": loss_cosine,
        "delta_mse": loss_delta,
        "proto_loss": loss_proto,
        "recon_mae": loss_recon,
    }


def evaluate(args, model, autoencoder, codebook, loader, device):
    model.eval()
    autoencoder.eval()
    sums = {
        "total": 0.0,
        "pred_mse": 0.0,
        "pred_mae": 0.0,
        "latent_mse": 0.0,
        "cosine_loss": 0.0,
        "delta_mse": 0.0,
        "proto_loss": 0.0,
        "recon_mae": 0.0,
    }
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            total, metrics = compute_losses(args, autoencoder, model, codebook, batch_x, batch_y, device)
            bsz = batch_x.size(0)
            total_count += bsz
            for key in sums:
                sums[key] += metrics[key].item() * bsz

    return {key: value / max(total_count, 1) for key, value in sums.items()}


def save_checkpoint(path, args, model, epoch, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def train(args, device):
    if args.pred_len % args.proto_patch_len != 0:
        raise ValueError("pred_len must be divisible by proto_patch_len.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    codebook_np = fit_codebook(args, autoencoder, device)
    np.save(os.path.join(args.output_dir, "latent_proto_codebook.npy"), codebook_np)
    codebook = torch.from_numpy(codebook_np).float().to(device)

    model = LatentForecaster(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    best_path = os.path.join(args.output_dir, "best_latent_proto_regularized.pt")
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"Latent prototype regularized forecaster | model={args.model} "
        f"loss_mode={args.loss_mode} "
        f"K={args.codebook_size} patch_len={args.proto_patch_len} tau={args.proto_tau} "
        f"proto_distance={args.proto_distance} normalize_proto={args.normalize_proto_patches} "
        f"lambda_latent={args.lambda_latent} lambda_delta={args.lambda_delta} "
        f"lambda_proto={args.lambda_proto}",
        flush=True,
    )
    print(f"Codebook shape: {codebook_np.shape}; AE frozen=True; inference uses model output only", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "total": 0.0,
            "pred_mse": 0.0,
            "pred_mae": 0.0,
            "latent_mse": 0.0,
            "cosine_loss": 0.0,
            "delta_mse": 0.0,
            "proto_loss": 0.0,
            "recon_mae": 0.0,
        }
        total_count = 0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            total, metrics = compute_losses(args, autoencoder, model, codebook, batch_x, batch_y, device)

            optimizer.zero_grad()
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bsz = batch_x.size(0)
            total_count += bsz
            for key in sums:
                sums[key] += metrics[key].item() * bsz

        train_metrics = {key: value / max(total_count, 1) for key, value in sums.items()}
        val_metrics = evaluate(args, model, autoencoder, codebook, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch {epoch:03d} | "
            f"train pred {train_metrics['pred_mse']:.6f}/{train_metrics['pred_mae']:.6f} "
            f"latent {train_metrics['latent_mse']:.6f} "
            f"cos {train_metrics['cosine_loss']:.6f} "
            f"delta {train_metrics['delta_mse']:.6f} "
            f"proto {train_metrics['proto_loss']:.6f} | "
            f"val pred {val_metrics['pred_mse']:.6f}/{val_metrics['pred_mae']:.6f} "
            f"latent {val_metrics['latent_mse']:.6f} "
            f"cos {val_metrics['cosine_loss']:.6f} "
            f"delta {val_metrics['delta_mse']:.6f} "
            f"proto {val_metrics['proto_loss']:.6f}",
            flush=True,
        )

        metric = val_metrics[args.early_stop_metric]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, args, model, epoch, val_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = evaluate(args, model, autoencoder, codebook, val_loader, device)
    test_metrics = evaluate(args, model, autoencoder, codebook, test_loader, device)

    summary = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "codebook_path": os.path.join(args.output_dir, "latent_proto_codebook.npy"),
    }
    with open(os.path.join(args.output_dir, "latent_proto_regularized_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nLatent prototype regularized [test]")
    print(f"  pred MSE/MAE:  {test_metrics['pred_mse']:.6f} / {test_metrics['pred_mae']:.6f}")
    print(f"  latent MSE:    {test_metrics['latent_mse']:.6f}")
    print(f"  cosine loss:   {test_metrics['cosine_loss']:.6f}")
    print(f"  delta MSE:     {test_metrics['delta_mse']:.6f}")
    print(f"  proto loss:    {test_metrics['proto_loss']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="LatentTSF forecaster with prototype regularization loss")

    parser.add_argument("--output_dir", type=str, default="./checkpoints/latent_proto_regularized")
    parser.add_argument("--autoencoder_path", type=str, required=True)

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

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--proto_patch_len", type=int, default=16)
    parser.add_argument("--proto_tau", type=float, default=0.1)
    parser.add_argument("--normalize_proto_patches", action="store_true", default=False)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument(
        "--proto_distance",
        type=str,
        default="sq_l2_mean",
        choices=["sq_l2_mean", "l2_mean", "l2"],
    )
    parser.add_argument("--kmeans_max_samples", type=int, default=200000)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--kmeans_max_iter", type=int, default=200)
    parser.add_argument("--kmeans_n_init", type=int, default=3)

    parser.add_argument("--lambda_latent", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.1)
    parser.add_argument("--lambda_proto", type=float, default=0.01)
    parser.add_argument("--loss_mode", type=str, default="regularized", choices=["regularized", "official_latent"])
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)

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
        default="pred_mse",
        choices=["total", "pred_mse", "latent_mse", "cosine_loss", "delta_mse", "proto_loss"],
    )

    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(args, device)


if __name__ == "__main__":
    main()
