import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from my_utils import args_train, set_seed, acquire_device, get_data
from my_temporal_AE import get_temporal_autoencoder


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def load_temporal_ae(args, device):
    """
    Build TemporalAutoEncoder / TemporalCNN architecture and load pretrained checkpoint.
    Important: args must match the AE training config:
        ae_type, seq_len, enc_in, d_model, d_ff
    """
    assert args.autoencoder_path is not None, "Please provide --autoencoder_path"

    model = get_temporal_autoencoder(args).float().to(device)

    print(f"Loading AE checkpoint from: {args.autoencoder_path}")
    state_dict = torch.load(args.autoencoder_path, map_location=device)

    # Compatible with plain checkpoint or DataParallel checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    freeze_model(model)

    print("Loaded and frozen TemporalAutoEncoder.")
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

            # Use the final pred_len part as true future target
            y_target = batch_y[:, -args.pred_len:, :]

            # TemporalAE uses Linear(seq_len -> d_model), so input length must equal args.seq_len
            assert batch_x.shape[1] == args.seq_len, (
                f"batch_x length {batch_x.shape[1]} != args.seq_len {args.seq_len}"
            )

            assert y_target.shape[1] == args.seq_len, (
                f"TemporalAE expects input length {args.seq_len}, "
                f"but y_target length is {y_target.shape[1]}. "
                f"Please set --pred_len == --seq_len for this first experiment."
            )

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
    PCA 2D visualization of z_y clusters.
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
    Decode cluster centers back to observation space and plot their mean-channel curves.

    centers shape:
        [K, d_model * enc_in]

    all_zy shape:
        [N, d_model, enc_in]
    """
    try:
        center_latents = centers.reshape(K, *all_zy.shape[1:])
        center_latents_torch = torch.tensor(center_latents, dtype=torch.float32).to(device)

        with torch.no_grad():
            center_decoded = model.decode(center_latents_torch).detach().cpu().numpy()

        # center_decoded shape: [K, seq_len, enc_in]
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

        # Also plot each variable/channel separately for small enc_in
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
    Save labels, centers, metrics, PCA plots, decoded center plots.
    """
    print("\nStart KMeans++ clustering on z_y...")

    cluster_list = [4, 8, 16]

    N = all_zy.shape[0]
    zy_flat = all_zy.reshape(N, -1)

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

        # Decode cluster centers
        decode_and_plot_centers(args, model, device, centers, all_zy, K, save_dir)

    metrics_path = os.path.join(save_dir, "kmeanspp_metrics.txt")
    with open(metrics_path, "w") as f:
        f.writelines(metrics_lines)

    print(f"\nSaved clustering metrics: {metrics_path}")
    print("KMeans++ clustering finished.")


def main():
    args = args_train()
    set_seed(args.seed)
    device = acquire_device(args)

    # First experiment should use seq_len == pred_len for TemporalAE
    assert args.seq_len == args.pred_len, (
        f"For TemporalAutoEncoder extraction, please set --seq_len == --pred_len. "
        f"Got seq_len={args.seq_len}, pred_len={args.pred_len}."
    )

    save_dir = (
        f"./latent_outputs/"
        f"{args.data}_sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_dff{args.d_ff}_{args.ae_type}"
    )

    model = load_temporal_ae(args, device)
    all_zx, all_zy, all_y = extract_latents(args, model, device, save_dir)
    run_kmeanspp(args, model, device, all_zy, save_dir)

    if args.gpu_type == "mps":
        torch.backends.mps.empty_cache()
    elif args.gpu_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()