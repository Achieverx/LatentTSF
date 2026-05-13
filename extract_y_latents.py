print(">>> [0] script import started", flush=True)

import os
import argparse
import random
import numpy as np

print(">>> [0.1] basic imports done", flush=True)

import matplotlib
matplotlib.use("Agg")   # 防止 Windows 后端卡住
import matplotlib.pyplot as plt

print(">>> [0.2] matplotlib imported", flush=True)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

print(">>> [0.3] sklearn imported", flush=True)

from data_provider.data_factory import data_provider
import torch
import torch.nn as nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def acquire_device(args):
    if args.use_gpu and args.gpu_type == "cuda" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Use GPU: cuda:{args.gpu}")
    elif args.use_gpu and args.gpu_type == "mps":
        device = torch.device("mps")
        print("Use GPU: mps")
    else:
        device = torch.device("cpu")
        print("Use CPU")
    return device


def get_data(args, flag):
    return data_provider(args, flag)


def args_train():
    parser = argparse.ArgumentParser(
        description="Extract future target latents and cluster z_y"
    )

    parser.add_argument("--task_name", type=str, required=True, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, required=True, default=0)
    parser.add_argument("--model_id", type=str, required=True, default="extract_zy_cluster")
    parser.add_argument("--model", type=str, required=True, default="DLinear")

    parser.add_argument("--data", type=str, required=True, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--ae_type", type=str, default="MLP", choices=["MLP"])
    parser.add_argument("--ae_loss", type=str, default="MSE", choices=["MSE", "MAE"])

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using GPU")
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
        print("Using cpu or mps")

    print("Args in experiment:")
    print(args)
    return args


class AutoEncoder(nn.Module):
    """
    Minimal MLP AutoEncoder for official LatentTSF MLP checkpoints.
    Input:  [B, T, enc_in]
    Latent: [B, T, d_model]
    Output: [B, T, enc_in]
    """
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(args.enc_in, args.d_ff),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.enc_in),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, x):
        return self.decode(self.encode(x))


def get_autoencoder(args):
    ae_type = getattr(args, "ae_type", "MLP").upper()
    if ae_type != "MLP":
        raise ValueError(
            f"This extraction script only supports official MLP AE checkpoints. Got ae_type={ae_type}"
        )

    print("Using minimal MLP AutoEncoder for official checkpoint", flush=True)
    print("  Latent shape: [batch, seq_len, d_model]", flush=True)
    return AutoEncoder(args)

print(">>> [0.4] project imports done", flush=True)


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def load_pretrained_mlp_ae(args, device):
    """
    Load pretrained MLP AutoEncoder from my_AE.py.

    This is for the official LatentTSF pretrained AE checkpoints:
        ae_type = MLP
        latent shape = [B, seq_len, d_model]

    Important:
        args.enc_in, args.d_model, args.d_ff, args.ae_type
        must match the checkpoint config.
    """
    assert args.autoencoder_path is not None, "Please provide --autoencoder_path"

    model = get_autoencoder(args).float().to(device)

    print(f"Loading pretrained AE checkpoint from: {args.autoencoder_path}")
    state_dict = torch.load(args.autoencoder_path, map_location=device)

    # Compatible with DataParallel checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    freeze_model(model)

    print("Loaded and frozen pretrained MLP AutoEncoder.")
    return model


def extract_latents(args, model, device, save_dir):
    """
    Extract:
        z_x = E(X)
        z_y = E(Y_future)

    Save:
        train_zx.npy
        train_zy.npy
        train_y.npy

    For MLP AE:
        batch_x:  [B, seq_len, enc_in]
        z_x:      [B, seq_len, d_model]

        y_target: [B, pred_len, enc_in]
        z_y:      [B, pred_len, d_model]
    """
    train_data, train_loader = get_data(args, flag="train")

    all_zx = []
    all_zy = []
    all_y = []

    print("\nStart extracting z_x and z_y from train split...")

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # true future target
            y_target = batch_y[:, -args.pred_len:, :]

            # MLP AE encodes last dimension enc_in -> d_model
            # It does not require pred_len == seq_len.
            z_x = model.encode(batch_x)
            z_y = model.encode(y_target)

            all_zx.append(z_x.detach().cpu().numpy())
            all_zy.append(z_y.detach().cpu().numpy())
            all_y.append(y_target.detach().cpu().numpy())

            if (i + 1) % 100 == 0:
                print(f"  extracted batches: {i + 1}/{len(train_loader)}")

    all_zx = np.concatenate(all_zx, axis=0)
    all_zy = np.concatenate(all_zy, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "train_zx.npy"), all_zx)
    np.save(os.path.join(save_dir, "train_zy.npy"), all_zy)
    np.save(os.path.join(save_dir, "train_y.npy"), all_y)

    print("\nSaved latent arrays:")
    print(f"  z_x shape: {all_zx.shape}")
    print(f"  z_y shape: {all_zy.shape}")
    print(f"  y shape:   {all_y.shape}")
    print(f"  save_dir:  {save_dir}")

    return all_zx, all_zy, all_y


def plot_pca(zy_flat, labels, K, save_dir):
    """
    PCA is only for visualization.
    KMeans is already done in high-dimensional z_y latent space.
    """
    pca = PCA(n_components=2, random_state=2021)
    zy_2d = pca.fit_transform(zy_flat)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        zy_2d[:, 0],
        zy_2d[:, 1],
        c=labels,
        s=5,
        alpha=0.7,
    )
    plt.title(f"KMeans++ on z_y latent space | K={K}")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.colorbar(scatter)
    plt.tight_layout()

    path = os.path.join(save_dir, f"zy_kmeanspp_pca_K{K}.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"  saved PCA plot: {path}")


def decode_and_plot_centers(args, model, device, centers, all_zy, K, save_dir):
    """
    Decode cluster centers back to original observation space.

    centers:
        [K, pred_len * d_model]

    all_zy:
        [N, pred_len, d_model]

    decoded centers:
        [K, pred_len, enc_in]
    """
    try:
        center_latents = centers.reshape(K, *all_zy.shape[1:])
        center_latents_torch = torch.tensor(center_latents, dtype=torch.float32).to(device)

        with torch.no_grad():
            center_decoded = model.decode(center_latents_torch).detach().cpu().numpy()

        np.save(
            os.path.join(save_dir, f"zy_kmeanspp_decoded_centers_K{K}.npy"),
            center_decoded,
        )

        # Plot channel-mean decoded center curves
        plt.figure(figsize=(8, 5))
        for k in range(K):
            curve = center_decoded[k].mean(axis=-1)
            plt.plot(curve, label=f"cluster {k}")

        plt.title(f"Decoded z_y cluster centers | K={K} | channel mean")
        plt.xlabel("future time step")
        plt.ylabel("decoded value, channel mean")
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = os.path.join(save_dir, f"zy_kmeanspp_decoded_centers_mean_K{K}.png")
        plt.savefig(path, dpi=200)
        plt.close()

        print(f"  saved decoded center curves: {path}")

        # Plot each channel separately if enc_in is not too large
        if center_decoded.shape[-1] <= 16:
            channel_dir = os.path.join(save_dir, f"decoded_centers_K{K}_channels")
            os.makedirs(channel_dir, exist_ok=True)

            enc_in = center_decoded.shape[-1]
            for c in range(enc_in):
                plt.figure(figsize=(8, 5))
                for k in range(K):
                    plt.plot(center_decoded[k, :, c], label=f"cluster {k}")
                plt.title(f"Decoded cluster centers | K={K} | channel {c}")
                plt.xlabel("future time step")
                plt.ylabel(f"decoded value, channel {c}")
                plt.legend(fontsize=8)
                plt.tight_layout()

                ch_path = os.path.join(channel_dir, f"channel_{c}.png")
                plt.savefig(ch_path, dpi=200)
                plt.close()

            print(f"  saved per-channel decoded center plots: {channel_dir}")

    except Exception as e:
        print(f"  decode cluster centers failed for K={K}: {e}")


def run_kmeanspp(args, model, device, all_zy, save_dir):
    """
    Run KMeans++ on flattened future latent trajectories z_y.

    Actual clustering happens on:
        all_zy.reshape(N, -1)

    PCA and decoder plots are only for visualization/interpretation.
    """
    print("\nStart KMeans++ clustering on z_y...")

    cluster_list = [8]

    N = all_zy.shape[0]
    zy_flat = all_zy.reshape(N, -1)

    zy_flat_norm = zy_flat - zy_flat.mean(axis=1, keepdims=True)
    zy_flat = zy_flat_norm / (zy_flat.std(axis=1, keepdims=True) + 1e-6)

    print(f"z_y original shape: {all_zy.shape}")
    print(f"z_y flat shape:     {zy_flat.shape}")

    metrics_lines = []

    for K in cluster_list:
        print(f"\n========== KMeans++ | K={K} ==========")

        kmeans = KMeans(
            n_clusters=K,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=2021,
        )

        labels = kmeans.fit_predict(zy_flat)
        centers = kmeans.cluster_centers_

        labels_path = os.path.join(save_dir, f"zy_kmeanspp_labels_K{K}.npy")
        centers_path = os.path.join(save_dir, f"zy_kmeanspp_centers_K{K}.npy")

        np.save(labels_path, labels)
        np.save(centers_path, centers)

        print(f"  saved labels:  {labels_path}")
        print(f"  saved centers: {centers_path}")

        # Cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        count_dict = dict(zip(unique.tolist(), counts.tolist()))

        print("Cluster counts:")
        for k in range(K):
            print(f"  cluster {k}: {count_dict.get(k, 0)}")

        print("Cluster ratios:")
        for k in range(K):
            ratio = count_dict.get(k, 0) / N
            print(f"  cluster {k}: {ratio:.4f}")

        # Metrics
        max_sil_samples = 5000
        if N > max_sil_samples:
            rng = np.random.default_rng(2021)
            idx = rng.choice(N, size=max_sil_samples, replace=False)
            sil = silhouette_score(zy_flat[idx], labels[idx])
            dbi = davies_bouldin_score(zy_flat[idx], labels[idx])
            metric_note = f"sampled_{max_sil_samples}"
        else:
            sil = silhouette_score(zy_flat, labels)
            dbi = davies_bouldin_score(zy_flat, labels)
            metric_note = "full"

        print(f"Silhouette score ({metric_note}): {sil:.4f}")
        print(f"Davies-Bouldin index ({metric_note}): {dbi:.4f}")

        metrics_lines.append(f"K={K}\n")
        metrics_lines.append(f"silhouette_{metric_note}: {sil:.6f}\n")
        metrics_lines.append(f"davies_bouldin_{metric_note}: {dbi:.6f}\n")
        metrics_lines.append("counts: " + str(count_dict) + "\n")
        metrics_lines.append("\n")

        # Visualization
        plot_pca(zy_flat, labels, K, save_dir)

        # Decode cluster centers for interpretation
        decode_and_plot_centers(args, model, device, centers, all_zy, K, save_dir)

    metrics_path = os.path.join(save_dir, "kmeanspp_metrics.txt")
    with open(metrics_path, "w") as f:
        f.writelines(metrics_lines)

    print(f"\nSaved clustering metrics: {metrics_path}")
    print("KMeans++ clustering finished.")


def main():
    print(">>> [1] before args_train", flush=True)
    args = args_train()

    print(">>> [2] after args_train", flush=True)
    set_seed(args.seed)

    # Windows 先强制 num_workers=0，避免 DataLoader 卡住
    args.num_workers = 0

    print(">>> [3] before acquire_device", flush=True)
    device = acquire_device(args)
    print(f">>> [4] device = {device}", flush=True)

    args.ae_type = getattr(args, "ae_type", "MLP")
    if args.ae_type.upper() != "MLP":
        print(f"Warning: official checkpoints are usually MLP. Current ae_type={args.ae_type}", flush=True)

    save_dir = (
        f"./latent_outputs/"
        f"{args.data}_sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_dff{args.d_ff}_{args.ae_type}"
    )

    print(">>> [5] before load_pretrained_mlp_ae", flush=True)
    model = load_pretrained_mlp_ae(args, device)
    print(">>> [6] after load_pretrained_mlp_ae", flush=True)

    print(">>> [7] before extract_latents", flush=True)
    all_zx, all_zy, all_y = extract_latents(args, model, device, save_dir)
    print(">>> [8] after extract_latents", flush=True)

    print(">>> [9] before run_kmeanspp", flush=True)
    run_kmeanspp(args, model, device, all_zy, save_dir)
    print(">>> [10] after run_kmeanspp", flush=True)

    if args.gpu_type == "mps":
        torch.backends.mps.empty_cache()
    elif args.gpu_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print(">>> extract_y_latents.py started")
    main()
