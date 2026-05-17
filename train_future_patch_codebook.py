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
from torch.utils.data import DataLoader, TensorDataset

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


def collect_split(args, autoencoder, flag, device):
    loader = official_loader(args, flag, shuffle=False)
    z_x_all, z_y_all, y_all = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().to(device)
            y = batch_y[:, -args.pred_len:, :].float().to(device)
            z_x = autoencoder.encode(x)
            z_y = autoencoder.encode(y)
            z_x_all.append(z_x.cpu())
            z_y_all.append(z_y.cpu())
            y_all.append(y.cpu())

    return {
        "z_x": torch.cat(z_x_all, dim=0).numpy().astype(np.float32),
        "z_y": torch.cat(z_y_all, dim=0).numpy().astype(np.float32),
        "y": torch.cat(y_all, dim=0).numpy().astype(np.float32),
    }


def patchify(seq, patch_len):
    if seq.shape[1] % patch_len != 0:
        raise ValueError(f"length={seq.shape[1]} must be divisible by patch_len={patch_len}.")
    return seq.reshape(seq.shape[0], seq.shape[1] // patch_len, patch_len * seq.shape[2])


def unpatchify(patches, seq_len, channels):
    return patches.reshape(patches.shape[0], seq_len, channels)


def normalize_patches_np(patches, eps):
    mean = patches.mean(axis=-1, keepdims=True)
    std = patches.std(axis=-1, keepdims=True)
    std = np.maximum(std, eps)
    return (patches - mean) / std, mean, std


def normalize_patches_torch(patches, eps):
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (patches - mean) / std, mean, std


def fit_codebook(args, train_z_y):
    patches = patchify(train_z_y, args.latent_patch_len)
    norm_patches, _, _ = normalize_patches_np(patches, args.norm_eps)
    flat = norm_patches.reshape(-1, norm_patches.shape[-1])

    sample_count = min(args.kmeans_max_samples, flat.shape[0])
    if sample_count < flat.shape[0]:
        rng = np.random.default_rng(args.seed)
        fit_data = flat[rng.choice(flat.shape[0], size=sample_count, replace=False)]
    else:
        fit_data = flat

    kmeans = MiniBatchKMeans(
        n_clusters=args.codebook_size,
        init="k-means++",
        batch_size=args.kmeans_batch_size,
        max_iter=args.kmeans_max_iter,
        n_init=args.kmeans_n_init,
        random_state=args.seed,
        verbose=0,
    )
    kmeans.fit(fit_data)
    return kmeans.cluster_centers_.astype(np.float32), kmeans


def assign_tokens(kmeans, z_y, patch_len, eps):
    patches = patchify(z_y, patch_len)
    norm_patches, _, _ = normalize_patches_np(patches, eps)
    labels = kmeans.predict(norm_patches.reshape(-1, norm_patches.shape[-1]))
    return labels.reshape(norm_patches.shape[0], norm_patches.shape[1]).astype(np.int64)


def assign_random_tokens(args, z_y, flag):
    num_patches = z_y.shape[1] // args.latent_patch_len
    rng = np.random.default_rng(args.seed + {"train": 0, "val": 1, "test": 2}[flag])
    return rng.integers(0, args.codebook_size, size=(z_y.shape[0], num_patches), dtype=np.int64)


def make_loader(split, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split["z_x"]),
            torch.from_numpy(split["z_y"]),
            torch.from_numpy(split["y"]),
            torch.from_numpy(split["tokens"]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


class LatentBaselineForecaster(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone_args = SimpleNamespace(**vars(args))
        backbone_args.enc_in = args.d_model
        backbone_args.dec_in = args.d_model
        backbone_args.c_out = args.d_model
        self.pred_len = args.pred_len
        self.backbone = model_dict[args.model].Model(backbone_args).float()

    def forward(self, z_x):
        z_base = self.backbone(z_x, None, None, None)
        return z_base[:, -self.pred_len:, :]


class LatentPatchCodebookComposer(torch.nn.Module):
    def __init__(self, args, codebook):
        super().__init__()
        self.pred_len = args.pred_len
        self.latent_patch_len = args.latent_patch_len
        self.num_patches = args.pred_len // args.latent_patch_len
        self.d_model = args.d_model
        self.codebook_size = args.codebook_size
        self.composition_mode = args.composition_mode
        self.softmax_temperature = args.softmax_temperature
        self.norm_eps = args.norm_eps

        self.baseline = LatentBaselineForecaster(args)
        self.context_proj = torch.nn.Linear(args.d_model, args.hidden_dim)
        self.base_patch_proj = torch.nn.Linear(args.latent_patch_len * args.d_model, args.hidden_dim)
        self.future_queries = torch.nn.Parameter(torch.randn(self.num_patches, args.hidden_dim) * 0.02)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=args.decoder_layers)
        self.token_head = torch.nn.Linear(args.hidden_dim, args.codebook_size)

        self.residual_head = torch.nn.Sequential(
            torch.nn.LayerNorm(args.hidden_dim),
            torch.nn.Linear(args.hidden_dim, args.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.hidden_dim, args.latent_patch_len * args.d_model),
        )
        self.alpha_head = torch.nn.Sequential(torch.nn.LayerNorm(args.hidden_dim), torch.nn.Linear(args.hidden_dim, 1))
        torch.nn.init.constant_(self.alpha_head[-1].bias, args.residual_alpha_init)
        self.beta_head = torch.nn.Sequential(torch.nn.LayerNorm(args.hidden_dim), torch.nn.Linear(args.hidden_dim, 1))
        torch.nn.init.constant_(self.beta_head[-1].bias, args.codebook_beta_init)

        self.register_buffer("codebook", torch.from_numpy(codebook).float())

    def compose_proto(self, logits, z_base, hard_decode):
        base_patches = patchify(z_base, self.latent_patch_len)
        _, base_mean, base_std = normalize_patches_torch(base_patches, self.norm_eps)

        if hard_decode:
            token_ids = logits.argmax(dim=-1)
            shape = self.codebook[token_ids]
        else:
            probs = torch.softmax(logits / self.softmax_temperature, dim=-1)
            shape = torch.einsum("bpk,kd->bpd", probs, self.codebook)

        proto_patches = shape * base_std + base_mean
        return unpatchify(proto_patches, self.pred_len, self.d_model)

    def forward(self, z_x, hard_decode=False):
        z_base = self.baseline(z_x)
        base_patches = patchify(z_base, self.latent_patch_len)
        memory = self.context_proj(torch.cat([z_x, z_base], dim=1))
        queries = self.future_queries.unsqueeze(0).expand(z_x.size(0), -1, -1)
        queries = queries + self.base_patch_proj(base_patches)
        h = self.decoder(queries, memory)

        logits = self.token_head(h)
        z_proto = self.compose_proto(logits, z_base, hard_decode=hard_decode)
        residual = unpatchify(self.residual_head(h), self.pred_len, self.d_model)
        pooled = memory.mean(dim=1)
        alpha = torch.sigmoid(self.alpha_head(pooled)).view(z_x.size(0), 1, 1)
        beta = torch.sigmoid(self.beta_head(pooled)).view(z_x.size(0), 1, 1)
        z_code_refine = z_base + beta * (z_proto - z_base)

        if self.composition_mode == "residual_only":
            z_hat = z_base + alpha * residual
        elif self.composition_mode == "codebook_residual":
            z_hat = z_code_refine + alpha * residual
        else:
            z_hat = z_code_refine

        return {
            "logits": logits,
            "z_base": z_base,
            "z_proto": z_proto,
            "z_code_refine": z_code_refine,
            "z_hat": z_hat,
            "residual": residual,
            "alpha": alpha,
            "beta": beta,
        }


def oracle_proto_from_tokens(args, codebook, z_y, tokens):
    patches = patchify(z_y, args.latent_patch_len)
    _, mean, std = normalize_patches_torch(patches, args.norm_eps)
    proto_shape = codebook[tokens]
    proto_patches = proto_shape * std + mean
    return unpatchify(proto_patches, args.pred_len, args.d_model)


def decode(autoencoder, z):
    return autoencoder.decode(z)


def mse_mae(pred, target):
    return F.mse_loss(pred, target), (pred - target).abs().mean()


def compute_losses(args, model, autoencoder, codebook, z_x, z_y, y, tokens):
    out = model(z_x, hard_decode=False)
    y_hat = decode(autoencoder, out["z_hat"])
    y_base = decode(autoencoder, out["z_base"])
    y_proto = decode(autoencoder, out["z_proto"])

    token_loss = F.cross_entropy(out["logits"].reshape(-1, args.codebook_size), tokens.reshape(-1))
    pred_mse, pred_mae = mse_mae(y_hat, y)
    latent_mse, latent_mae = mse_mae(out["z_hat"], z_y)
    base_mse, base_mae = mse_mae(y_base, y)
    proto_mse, proto_mae = mse_mae(y_proto, y)
    proto_latent_mse, proto_latent_mae = mse_mae(out["z_proto"], z_y)

    if args.composition_mode == "residual_only":
        residual_target = z_y - out["z_base"].detach()
    else:
        residual_target = z_y - out["z_proto"].detach()
    residual_mse, _ = mse_mae(out["residual"], residual_target)

    total = pred_mse + args.lambda_baseline * base_mse
    if args.composition_mode != "residual_only":
        total = total + args.lambda_token * token_loss
    if args.composition_mode != "codebook":
        total = total + args.lambda_residual * residual_mse
    if args.lambda_latent > 0:
        total = total + args.lambda_latent * latent_mse

    token_acc = (out["logits"].argmax(dim=-1) == tokens).float().mean()

    return total, {
        "loss": total,
        "token_ce": token_loss,
        "token_acc": token_acc,
        "pred_mse": pred_mse,
        "pred_mae": pred_mae,
        "latent_mse": latent_mse,
        "latent_mae": latent_mae,
        "baseline_mse": base_mse,
        "baseline_mae": base_mae,
        "soft_proto_mse": proto_mse,
        "soft_proto_mae": proto_mae,
        "soft_proto_latent_mse": proto_latent_mse,
        "soft_proto_latent_mae": proto_latent_mae,
        "residual_mse": residual_mse,
        "alpha": out["alpha"].mean(),
        "beta": out["beta"].mean(),
    }


def evaluate(args, model, autoencoder, codebook, loader, device):
    model.eval()
    sums = {
        "loss": 0.0,
        "token_ce": 0.0,
        "token_acc": 0.0,
        "pred_mse": 0.0,
        "pred_mae": 0.0,
        "latent_mse": 0.0,
        "baseline_mse": 0.0,
        "baseline_mae": 0.0,
        "soft_proto_mse": 0.0,
        "soft_proto_mae": 0.0,
        "soft_proto_latent_mse": 0.0,
        "hard_proto_mse": 0.0,
        "hard_proto_mae": 0.0,
        "hard_proto_latent_mse": 0.0,
        "oracle_proto_mse": 0.0,
        "oracle_proto_mae": 0.0,
        "oracle_proto_latent_mse": 0.0,
        "oracle_proto_latent_mae": 0.0,
        "residual_mse": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
    }
    total_count = 0

    with torch.no_grad():
        for z_x, z_y, y, tokens in loader:
            z_x = z_x.float().to(device)
            z_y = z_y.float().to(device)
            y = y.float().to(device)
            tokens = tokens.long().to(device)

            _, metrics = compute_losses(args, model, autoencoder, codebook, z_x, z_y, y, tokens)
            hard_out = model(z_x, hard_decode=True)
            y_hard = decode(autoencoder, hard_out["z_proto"])
            z_oracle = oracle_proto_from_tokens(args, codebook, z_y, tokens)
            y_oracle = decode(autoencoder, z_oracle)

            hard_mse, hard_mae = mse_mae(y_hard, y)
            hard_latent_mse, _ = mse_mae(hard_out["z_proto"], z_y)
            oracle_mse, oracle_mae = mse_mae(y_oracle, y)
            oracle_latent_mse, oracle_latent_mae = mse_mae(z_oracle, z_y)

            bsz = z_x.size(0)
            total_count += bsz
            for key in [
                "loss",
                "token_ce",
                "token_acc",
                "pred_mse",
                "pred_mae",
                "latent_mse",
                "baseline_mse",
                "baseline_mae",
                "soft_proto_mse",
                "soft_proto_mae",
                "soft_proto_latent_mse",
                "residual_mse",
                "alpha",
                "beta",
            ]:
                sums[key] += metrics[key].item() * bsz
            sums["hard_proto_mse"] += hard_mse.item() * bsz
            sums["hard_proto_mae"] += hard_mae.item() * bsz
            sums["hard_proto_latent_mse"] += hard_latent_mse.item() * bsz
            sums["oracle_proto_mse"] += oracle_mse.item() * bsz
            sums["oracle_proto_mae"] += oracle_mae.item() * bsz
            sums["oracle_proto_latent_mse"] += oracle_latent_mse.item() * bsz
            sums["oracle_proto_latent_mae"] += oracle_latent_mae.item() * bsz

    return {key: value / max(total_count, 1) for key, value in sums.items()}


def parse_beta_values(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def oracle_interpolation_sweep(args, model, autoencoder, codebook, loader, device):
    model.eval()
    betas = parse_beta_values(args.oracle_betas)
    sums = {
        beta: {
            "mse": 0.0,
            "mae": 0.0,
            "latent_mse": 0.0,
            "latent_mae": 0.0,
        }
        for beta in betas
    }
    total_count = 0

    with torch.no_grad():
        for z_x, z_y, y, tokens in loader:
            z_x = z_x.float().to(device)
            z_y = z_y.float().to(device)
            y = y.float().to(device)
            tokens = tokens.long().to(device)

            z_base = model.baseline(z_x)
            z_oracle = oracle_proto_from_tokens(args, codebook, z_y, tokens)

            bsz = z_x.size(0)
            total_count += bsz
            for beta in betas:
                z_interp = z_base + beta * (z_oracle - z_base)
                y_interp = decode(autoencoder, z_interp)
                obs_mse, obs_mae = mse_mae(y_interp, y)
                latent_mse, latent_mae = mse_mae(z_interp, z_y)
                sums[beta]["mse"] += obs_mse.item() * bsz
                sums[beta]["mae"] += obs_mae.item() * bsz
                sums[beta]["latent_mse"] += latent_mse.item() * bsz
                sums[beta]["latent_mae"] += latent_mae.item() * bsz

    return {
        str(beta): {key: value / max(total_count, 1) for key, value in metrics.items()}
        for beta, metrics in sums.items()
    }


def print_oracle_sweep(name, sweep):
    print(f"\nOracle code interpolation sweep [{name}]")
    best_beta = None
    best_mse = float("inf")
    for beta_text, metrics in sweep.items():
        mse = metrics["mse"]
        if mse < best_mse:
            best_mse = mse
            best_beta = beta_text
        print(
            f"  beta={float(beta_text):.3f} "
            f"MSE/MAE={metrics['mse']:.6f}/{metrics['mae']:.6f} "
            f"latent={metrics['latent_mse']:.6f}/{metrics['latent_mae']:.6f}",
            flush=True,
        )
    print(f"  best beta={float(best_beta):.3f} MSE={best_mse:.6f}", flush=True)


def pretrain_baseline(args, baseline, autoencoder, train_loader, val_loader, device):
    if args.baseline_pretrain_epochs <= 0:
        return []

    optimizer = torch.optim.AdamW(baseline.parameters(), lr=args.baseline_lr, weight_decay=args.weight_decay)
    history = []
    best_state = None
    best_metric = float("inf")
    bad_epochs = 0

    print(
        f"\nPretraining latent baseline | epochs={args.baseline_pretrain_epochs} "
        f"lr={args.baseline_lr} lambda_latent={args.baseline_lambda_latent}",
        flush=True,
    )

    for epoch in range(1, args.baseline_pretrain_epochs + 1):
        baseline.train()
        sums = {"obs_mse": 0.0, "obs_mae": 0.0, "latent_mse": 0.0}
        total_count = 0

        for z_x, z_y, y, tokens in train_loader:
            z_x = z_x.float().to(device)
            z_y = z_y.float().to(device)
            y = y.float().to(device)

            z_base = baseline(z_x)
            y_base = decode(autoencoder, z_base)
            obs_mse, obs_mae = mse_mae(y_base, y)
            latent_mse, _ = mse_mae(z_base, z_y)
            loss = obs_mse + args.baseline_lambda_latent * latent_mse

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), args.grad_clip)
            optimizer.step()

            bsz = z_x.size(0)
            total_count += bsz
            sums["obs_mse"] += obs_mse.item() * bsz
            sums["obs_mae"] += obs_mae.item() * bsz
            sums["latent_mse"] += latent_mse.item() * bsz

        train_metrics = {key: value / max(total_count, 1) for key, value in sums.items()}
        val_metrics = evaluate_baseline(args, baseline, autoencoder, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"baseline epoch {epoch:03d} | "
            f"train {train_metrics['obs_mse']:.6f}/{train_metrics['obs_mae']:.6f} "
            f"latent {train_metrics['latent_mse']:.6f} | "
            f"val {val_metrics['obs_mse']:.6f}/{val_metrics['obs_mae']:.6f} "
            f"latent {val_metrics['latent_mse']:.6f}",
            flush=True,
        )

        metric = val_metrics["obs_mse"]
        if metric < best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in baseline.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.baseline_pretrain_patience:
                print(f"Baseline early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        baseline.load_state_dict(best_state)
    return history


def evaluate_baseline(args, baseline, autoencoder, loader, device):
    baseline.eval()
    sums = {"obs_mse": 0.0, "obs_mae": 0.0, "latent_mse": 0.0, "latent_mae": 0.0}
    total_count = 0
    with torch.no_grad():
        for z_x, z_y, y, tokens in loader:
            z_x = z_x.float().to(device)
            z_y = z_y.float().to(device)
            y = y.float().to(device)
            z_base = baseline(z_x)
            y_base = decode(autoencoder, z_base)
            obs_mse, obs_mae = mse_mae(y_base, y)
            latent_mse, latent_mae = mse_mae(z_base, z_y)
            bsz = z_x.size(0)
            total_count += bsz
            sums["obs_mse"] += obs_mse.item() * bsz
            sums["obs_mae"] += obs_mae.item() * bsz
            sums["latent_mse"] += latent_mse.item() * bsz
            sums["latent_mae"] += latent_mae.item() * bsz
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
    if args.pred_len % args.latent_patch_len != 0:
        raise ValueError("pred_len must be divisible by latent_patch_len.")
    if args.hidden_dim % args.n_heads != 0:
        raise ValueError("hidden_dim must be divisible by n_heads.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    splits = {flag: collect_split(args, autoencoder, flag, device) for flag in ["train", "val", "test"]}
    codebook_np, kmeans = fit_codebook(args, splits["train"]["z_y"])
    np.save(os.path.join(args.output_dir, "latent_patch_shape_codebook.npy"), codebook_np)

    for flag, split in splits.items():
        if args.random_token_assignment:
            split["tokens"] = assign_random_tokens(args, split["z_y"], flag)
        else:
            split["tokens"] = assign_tokens(kmeans, split["z_y"], args.latent_patch_len, args.norm_eps)
        np.save(os.path.join(args.output_dir, f"{flag}_latent_patch_tokens.npy"), split["tokens"])

    train_loader = make_loader(splits["train"], args.batch_size, True)
    val_loader = make_loader(splits["val"], args.batch_size, False)
    test_loader = make_loader(splits["test"], args.batch_size, False)

    model = LatentPatchCodebookComposer(args, codebook_np).to(device)
    codebook = model.codebook
    baseline_pretrain_history = pretrain_baseline(
        args,
        model.baseline,
        autoencoder,
        train_loader,
        val_loader,
        device,
    )
    if args.baseline_pretrain_epochs > 0 and not args.train_baseline_during_refine:
        for param in model.baseline.parameters():
            param.requires_grad = False
        model.baseline.eval()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_path = os.path.join(args.output_dir, "best_latent_patch_codebook.pt")

    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    print(
        f"Latent patch codebook | model={args.model} mode={args.composition_mode} "
        f"K={args.codebook_size} latent_patch_len={args.latent_patch_len} "
        f"num_patches={args.pred_len // args.latent_patch_len} "
        f"random_tokens={args.random_token_assignment}",
        flush=True,
    )
    print(f"Codebook shape: {codebook_np.shape}; AE frozen=True", flush=True)

    val_oracle_sweep = oracle_interpolation_sweep(args, model, autoencoder, codebook, val_loader, device)
    test_oracle_sweep = oracle_interpolation_sweep(args, model, autoencoder, codebook, test_loader, device)
    print_oracle_sweep("val", val_oracle_sweep)
    print_oracle_sweep("test", test_oracle_sweep)

    if args.oracle_only:
        summary = {
            "val_oracle_interpolation_sweep": val_oracle_sweep,
            "test_oracle_interpolation_sweep": test_oracle_sweep,
            "codebook_path": os.path.join(args.output_dir, "latent_patch_shape_codebook.npy"),
            "random_token_assignment": args.random_token_assignment,
            "composition_mode": args.composition_mode,
            "baseline_pretrain_history": baseline_pretrain_history,
        }
        with open(os.path.join(args.output_dir, "oracle_code_interpolation_sweep.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        autoencoder.eval()
        sums = {
            "loss": 0.0,
            "token_ce": 0.0,
            "token_acc": 0.0,
            "pred_mse": 0.0,
            "pred_mae": 0.0,
            "baseline_mse": 0.0,
            "soft_proto_mse": 0.0,
            "residual_mse": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
        }
        total_count = 0

        for z_x, z_y, y, tokens in train_loader:
            z_x = z_x.float().to(device)
            z_y = z_y.float().to(device)
            y = y.float().to(device)
            tokens = tokens.long().to(device)

            loss, metrics = compute_losses(args, model, autoencoder, codebook, z_x, z_y, y, tokens)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bsz = z_x.size(0)
            total_count += bsz
            for key in sums:
                sums[key] += metrics[key].item() * bsz

        train_metrics = {key: value / max(total_count, 1) for key, value in sums.items()}
        val_metrics = evaluate(args, model, autoencoder, codebook, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch {epoch:03d} | "
            f"train pred {train_metrics['pred_mse']:.6f}/{train_metrics['pred_mae']:.6f} "
            f"base {train_metrics['baseline_mse']:.6f} soft {train_metrics['soft_proto_mse']:.6f} "
            f"ce {train_metrics['token_ce']:.4f} acc {train_metrics['token_acc']:.4f} "
            f"beta {train_metrics['beta']:.4f} alpha {train_metrics['alpha']:.4f} | "
            f"val pred {val_metrics['pred_mse']:.6f}/{val_metrics['pred_mae']:.6f} "
            f"base {val_metrics['baseline_mse']:.6f}/{val_metrics['baseline_mae']:.6f} "
            f"soft {val_metrics['soft_proto_mse']:.6f} "
            f"hard {val_metrics['hard_proto_mse']:.6f} "
            f"oracle {val_metrics['oracle_proto_mse']:.6f} "
            f"ce {val_metrics['token_ce']:.4f} acc {val_metrics['token_acc']:.4f} "
            f"beta {val_metrics['beta']:.4f} alpha {val_metrics['alpha']:.4f}",
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
        "codebook_path": os.path.join(args.output_dir, "latent_patch_shape_codebook.npy"),
        "random_token_assignment": args.random_token_assignment,
        "composition_mode": args.composition_mode,
        "baseline_pretrain_history": baseline_pretrain_history,
        "val_oracle_interpolation_sweep": val_oracle_sweep,
        "test_oracle_interpolation_sweep": test_oracle_sweep,
    }
    with open(os.path.join(args.output_dir, "latent_patch_codebook_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nLatent patch codebook [test]")
    print(f"  pred MSE/MAE:        {test_metrics['pred_mse']:.6f} / {test_metrics['pred_mae']:.6f}")
    print(f"  baseline MSE/MAE:    {test_metrics['baseline_mse']:.6f} / {test_metrics['baseline_mae']:.6f}")
    print(f"  soft proto MSE/MAE:  {test_metrics['soft_proto_mse']:.6f} / {test_metrics['soft_proto_mae']:.6f}")
    print(f"  hard proto MSE/MAE:  {test_metrics['hard_proto_mse']:.6f} / {test_metrics['hard_proto_mae']:.6f}")
    print(f"  oracle MSE/MAE:      {test_metrics['oracle_proto_mse']:.6f} / {test_metrics['oracle_proto_mae']:.6f}")
    print(f"  token CE/acc:        {test_metrics['token_ce']:.6f} / {test_metrics['token_acc']:.6f}")
    print(f"  latent oracle MSE:   {test_metrics['oracle_proto_latent_mse']:.6f}")
    print(f"  beta:                {test_metrics['beta']:.6f}")
    print(f"  alpha:               {test_metrics['alpha']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="LatentTSF baseline + latent future patch codebook composition")

    parser.add_argument("--output_dir", type=str, default="./checkpoints/latent_patch_codebook")
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
    parser.add_argument("--latent_patch_len", type=int, default=16)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)

    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--random_token_assignment", action="store_true", default=False)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--kmeans_max_samples", type=int, default=200000)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--kmeans_max_iter", type=int, default=200)
    parser.add_argument("--kmeans_n_init", type=int, default=3)

    parser.add_argument(
        "--composition_mode",
        type=str,
        default="codebook",
        choices=["codebook", "codebook_residual", "residual_only"],
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument("--codebook_beta_init", type=float, default=-5.0)
    parser.add_argument("--residual_alpha_init", type=float, default=-5.0)
    parser.add_argument("--oracle_betas", type=str, default="0.0,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--oracle_only", action="store_true", default=False)

    parser.add_argument("--lambda_token", type=float, default=0.1)
    parser.add_argument("--lambda_residual", type=float, default=0.1)
    parser.add_argument("--lambda_baseline", type=float, default=0.1)
    parser.add_argument("--lambda_latent", type=float, default=0.0)
    parser.add_argument("--baseline_pretrain_epochs", type=int, default=0)
    parser.add_argument("--baseline_pretrain_patience", type=int, default=8)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--baseline_lambda_latent", type=float, default=0.1)
    parser.add_argument("--train_baseline_during_refine", action="store_true", default=False)

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
        choices=[
            "loss",
            "pred_mse",
            "baseline_mse",
            "soft_proto_mse",
            "hard_proto_mse",
            "oracle_proto_mse",
            "token_ce",
        ],
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
