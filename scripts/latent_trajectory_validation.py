import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder


DEFAULT_ETTH1_AE = (
    "./checkpoints/"
    "AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0/"
    "checkpoint.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate whether AE latent trajectory displacements contain retrieval signal."
    )
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--autoencoder_path", type=str, default=DEFAULT_ETTH1_AE)
    parser.add_argument("--pool", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means use all train samples.")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--smooth_pairs", type=int, default=50000)
    parser.add_argument("--tsne_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", type=str, default="./results/latent_trajectory_validation/ETTh1")
    return parser.parse_args()


def build_data_args(args):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data=args.data,
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        freq=args.freq,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        step=args.step,
        seasonal_patterns="Monthly",
        embed="timeF",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_ratio=0,
    )


def build_ae_args(args):
    return SimpleNamespace(
        seq_len=args.seq_len,
        enc_in=args.enc_in,
        d_model=args.d_model,
        d_ff=args.d_ff,
        ae_type=args.ae_type,
        revin_affine=1,
    )


def select_device(name):
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def load_autoencoder(args, device):
    if not os.path.exists(args.autoencoder_path):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {args.autoencoder_path}")

    model = get_autoencoder(build_ae_args(args)).float().to(device)
    try:
        state = torch.load(args.autoencoder_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.autoencoder_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def pool_latent(z, mode):
    if mode == "last":
        return z[:, -1, :]
    if mode == "mean":
        return z.mean(dim=1)
    raise ValueError(f"Unsupported pool mode: {mode}")


def encode_dataset(args, model, loader, device):
    pooled_x, pooled_y, futures = [], [], []
    seen = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in tqdm(loader, desc="Encoding train split"):
            batch_x = batch_x.float().to(device)
            batch_future = batch_y[:, -args.pred_len:, :].float().to(device)

            z_x = model.encode(batch_x)
            z_y = model.encode(batch_future)

            pooled_x.append(pool_latent(z_x, args.pool).cpu().numpy())
            pooled_y.append(pool_latent(z_y, args.pool).cpu().numpy())
            futures.append(batch_future.cpu().numpy())

            seen += batch_x.shape[0]
            if args.max_samples and seen >= args.max_samples:
                break

    z_x_pool = np.concatenate(pooled_x, axis=0)
    z_y_pool = np.concatenate(pooled_y, axis=0)
    y_future = np.concatenate(futures, axis=0)

    if args.max_samples:
        z_x_pool = z_x_pool[:args.max_samples]
        z_y_pool = z_y_pool[:args.max_samples]
        y_future = y_future[:args.max_samples]

    d = z_y_pool - z_x_pool
    return d.astype(np.float32), y_future.astype(np.float32)


def future_descriptors(y):
    t = np.linspace(-1.0, 1.0, y.shape[1], dtype=np.float32)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)
    slope = ((y - y.mean(axis=1, keepdims=True)) * t_centered[None, :, None]).sum(axis=1) / denom
    slope = slope.mean(axis=1)

    trend = y[:, -1, :].mean(axis=1) - y[:, 0, :].mean(axis=1)
    volatility = y.std(axis=1).mean(axis=1)

    centered = y - y.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    if spectrum.shape[1] > 1:
        dom_freq = np.argmax(spectrum[:, 1:, :].mean(axis=2), axis=1) + 1
    else:
        dom_freq = np.zeros(y.shape[0], dtype=np.int64)

    return {
        "future_trend": trend.astype(np.float32),
        "future_slope": slope.astype(np.float32),
        "future_frequency": dom_freq.astype(np.float32),
        "future_volatility": volatility.astype(np.float32),
    }


def mse_to_neighbors(y_flat, neighbor_indices):
    anchor = y_flat[:, None, :]
    neighbor_y = y_flat[neighbor_indices]
    return ((anchor - neighbor_y) ** 2).mean(axis=2).mean(axis=1)


def random_neighbors(n, k, rng):
    idx = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        choices = rng.integers(0, n - 1, size=k)
        choices = choices + (choices >= i)
        idx[i] = choices
    return idx


def knn_consistency(d, y, k, seed):
    y_flat = y.reshape(y.shape[0], -1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(d)
    distances, indices = nn.kneighbors(d)
    latent_nn = indices[:, 1:]

    rng = np.random.default_rng(seed)
    rand_nn = random_neighbors(len(d), k, rng)

    future_mse = mse_to_neighbors(y_flat, latent_nn)
    random_mse = mse_to_neighbors(y_flat, rand_nn)

    oracle = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    oracle.fit(y_flat)
    _, oracle_indices = oracle.kneighbors(y_flat)
    oracle_nn = oracle_indices[:, 1:]
    oracle_mse = mse_to_neighbors(y_flat, oracle_nn)

    return {
        "latent_neighbor_future_mse": float(future_mse.mean()),
        "random_future_mse": float(random_mse.mean()),
        "oracle_future_mse": float(oracle_mse.mean()),
        "latent_vs_random_ratio": float(future_mse.mean() / random_mse.mean()),
        "oracle_vs_random_ratio": float(oracle_mse.mean() / random_mse.mean()),
        "latent_neighbor_distance": float(distances[:, 1:].mean()),
        "k": int(k),
    }


def smoothness_check(d, y, pair_count, seed):
    rng = np.random.default_rng(seed)
    n = len(d)
    i = rng.integers(0, n, size=pair_count)
    j = rng.integers(0, n - 1, size=pair_count)
    j = j + (j >= i)

    latent_dist = np.linalg.norm(d[i] - d[j], axis=1)
    future_dist = ((y[i] - y[j]) ** 2).mean(axis=(1, 2))

    pearson = pearsonr(latent_dist, future_dist)
    spearman = spearmanr(latent_dist, future_dist)
    return {
        "sampled_pairs": int(pair_count),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }, latent_dist, future_dist


def cluster_probe(d, seed):
    if len(d) < 20:
        return {"best_k": None, "best_silhouette": None}
    x = PCA(n_components=min(10, d.shape[1]), random_state=seed).fit_transform(d)
    best = {"best_k": None, "best_silhouette": -1.0}
    for k in range(2, min(10, len(d) - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(x)
        score = silhouette_score(x, labels)
        if score > best["best_silhouette"]:
            best = {"best_k": int(k), "best_silhouette": float(score)}
    return best


def save_scatter(points, colors, name, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    sc = ax.scatter(points[:, 0], points[:, 1], c=colors, s=5, cmap="viridis", alpha=0.75)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close(fig)


def visualize(d, descriptors, args):
    pca_points = PCA(n_components=2, random_state=args.seed).fit_transform(d)
    for label, values in descriptors.items():
        save_scatter(pca_points, values, f"pca_by_{label}", args.output_dir)

    tsne_n = min(args.tsne_samples, len(d))
    if tsne_n >= 10:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(d), size=tsne_n, replace=False)
        perplexity = min(30, max(5, (tsne_n - 1) // 3))
        tsne_points = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(d[idx])
        for label, values in descriptors.items():
            save_scatter(tsne_points, values[idx], f"tsne_by_{label}", args.output_dir)

    return pca_points


def save_smoothness_plot(latent_dist, future_dist, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(latent_dist, future_dist, s=3, alpha=0.25)
    ax.set_xlabel("||d_i - d_j||")
    ax.set_ylabel("MSE(y_i, y_j)")
    ax.set_title("local_smoothness")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "local_smoothness.png"))
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    model = load_autoencoder(args, device)
    _, train_loader = data_provider(build_data_args(args), "train")

    d, y = encode_dataset(args, model, train_loader, device)
    print(f"Encoded displacement shape: {d.shape}; future shape: {y.shape}")

    descriptors = future_descriptors(y)
    pca_points = visualize(d, descriptors, args)

    metrics = {
        "config": vars(args),
        "num_samples": int(len(d)),
        "displacement_dim": int(d.shape[1]),
        "pool": args.pool,
        "knn_consistency": knn_consistency(d, y, args.knn_k, args.seed),
        "cluster_probe": cluster_probe(d, args.seed),
    }
    smooth_metrics, latent_dist, future_dist = smoothness_check(d, y, args.smooth_pairs, args.seed)
    metrics["local_smoothness"] = smooth_metrics

    np.savez_compressed(
        os.path.join(args.output_dir, "latent_trajectory_arrays.npz"),
        displacement=d,
        future=y,
        pca=pca_points,
        **descriptors,
    )
    save_smoothness_plot(latent_dist, future_dist, args.output_dir)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
