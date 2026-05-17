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
from train_frozen_dlinear_branch_selector import FrozenDLinearMultiBranch, LatentForecaster


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
    if args.unfreeze_encoder:
        enc = autoencoder.module.encoder if hasattr(autoencoder, "module") else autoencoder.encoder
        for param in enc.parameters():
            param.requires_grad = True
    return autoencoder


def load_branch_oracle(args, device):
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
        raise RuntimeError(f"Could not load branch oracle. missing={missing}, unexpected={unexpected}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, branch_args


def load_latenttsf_baseline(args, device):
    if not args.latenttsf_checkpoint:
        return None
    checkpoint = torch.load(args.latenttsf_checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    base_args = SimpleNamespace(**vars(args))
    for key, value in ckpt_args.items():
        setattr(base_args, key, value)
    model = LatentForecaster(base_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_latenttsf_state(args, device):
    if not args.latenttsf_checkpoint:
        return None
    checkpoint = torch.load(args.latenttsf_checkpoint, map_location=device, weights_only=False)
    return checkpoint["model_state_dict"]


class BranchRegularizedForecaster(torch.nn.Module):
    def __init__(self, args, num_branches):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(args.d_model),
            torch.nn.Linear(args.d_model, args.adapter_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.adapter_hidden_dim, args.d_model),
        )
        torch.nn.init.zeros_(self.adapter[-1].weight)
        torch.nn.init.zeros_(self.adapter[-1].bias)
        self.forecaster = LatentForecaster(args)
        self.branch_head = torch.nn.Sequential(
            torch.nn.LayerNorm(args.d_model),
            torch.nn.Linear(args.d_model, args.branch_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.branch_hidden_dim, num_branches),
        )

    def forward(self, z_x):
        z = z_x + self.adapter(z_x)
        z_pred = self.forecaster(z)
        branch_logits = self.branch_head(z.detach().mean(dim=1))
        return z_pred, branch_logits

    def main_parameters(self):
        yield from self.adapter.parameters()
        yield from self.forecaster.parameters()

    def branch_parameters(self):
        yield from self.branch_head.parameters()


def decode_branches(autoencoder, z_branches):
    bsz, branches, steps, dim = z_branches.shape
    y_branches = autoencoder.decode(z_branches.reshape(bsz * branches, steps, dim))
    return y_branches.reshape(bsz, branches, steps, -1)


def delta_loss(z_pred, z_y):
    if z_pred.size(1) <= 1:
        return z_pred.new_zeros(())
    return F.mse_loss(z_pred[:, 1:] - z_pred[:, :-1], z_y[:, 1:] - z_y[:, :-1])


def make_branch_targets(args, autoencoder, branch_oracle, z_x, y_true):
    with torch.no_grad():
        z_base, z_branches = branch_oracle.forward_with_base(z_x)
        y_branches = decode_branches(autoencoder, z_branches)
        errors = ((y_branches - y_true[:, None]) ** 2).mean(dim=(2, 3))
        q = torch.softmax(-errors / args.branch_tau, dim=-1)
        labels = errors.argmin(dim=-1)
    return q, labels, errors


def encode_batch(args, autoencoder, batch_x, batch_y, device):
    x = batch_x.float().to(device)
    y = batch_y[:, -args.pred_len:, :].float().to(device)
    if args.unfreeze_encoder:
        z_x = autoencoder.encode(x)
    else:
        with torch.no_grad():
            z_x = autoencoder.encode(x)
    with torch.no_grad():
        z_y = autoencoder.encode(y)
    return y, z_x, z_y


def compute_losses(args, model, autoencoder, branch_oracle, batch_x, batch_y, device):
    y, z_x, z_y = encode_batch(args, autoencoder, batch_x, batch_y, device)
    q, labels, branch_errors = make_branch_targets(args, autoencoder, branch_oracle, z_x, y)
    z_pred, branch_logits = model(z_x)
    y_pred = autoencoder.decode(z_pred)

    loss_forecast = F.mse_loss(y_pred, y)
    loss_latent = F.mse_loss(z_pred, z_y)
    loss_delta = delta_loss(z_pred, z_y)
    loss_branch = F.kl_div(F.log_softmax(branch_logits, dim=-1), q, reduction="batchmean")
    main_total = (
        loss_forecast
        + args.lambda_latent * loss_latent
        + args.lambda_delta * loss_delta
    )

    pred_labels = branch_logits.argmax(dim=-1)
    top3 = branch_logits.topk(min(3, branch_logits.size(-1)), dim=-1).indices
    return main_total, loss_branch, {
        "total": main_total.detach(),
        "forecast_mse": loss_forecast.detach(),
        "forecast_mae": (y_pred - y).abs().mean().detach(),
        "latent_mse": loss_latent.detach(),
        "delta_mse": loss_delta.detach(),
        "branch_kl": loss_branch.detach(),
        "branch_winner_at_1": (pred_labels == labels).float().mean().detach(),
        "branch_winner_at_3": (top3 == labels[:, None]).any(dim=-1).float().mean().detach(),
        "oracle_best_mse": branch_errors.min(dim=-1).values.mean().detach(),
    }


def evaluate(args, model, autoencoder, branch_oracle, loader, device, latenttsf_baseline=None):
    model.eval()
    sums = {
        "total": 0.0,
        "forecast_mse": 0.0,
        "forecast_mae": 0.0,
        "latent_mse": 0.0,
        "delta_mse": 0.0,
        "branch_kl": 0.0,
        "branch_winner_at_1": 0.0,
        "branch_winner_at_3": 0.0,
        "oracle_best_mse": 0.0,
        "latenttsf_mse": 0.0,
        "latenttsf_mae": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            total, branch_loss, metrics = compute_losses(args, model, autoencoder, branch_oracle, batch_x, batch_y, device)
            bsz = batch_x.size(0)
            count += bsz
            for key in [
                "total",
                "forecast_mse",
                "forecast_mae",
                "latent_mse",
                "delta_mse",
                "branch_kl",
                "branch_winner_at_1",
                "branch_winner_at_3",
                "oracle_best_mse",
            ]:
                sums[key] += metrics[key].item() * bsz

            if latenttsf_baseline is not None:
                y, z_x, _ = encode_batch(args, autoencoder, batch_x, batch_y, device)
                y_base = autoencoder.decode(latenttsf_baseline(z_x))
                sums["latenttsf_mse"] += F.mse_loss(y_base, y, reduction="sum").item() / (
                    y.size(1) * y.size(2)
                )
                sums["latenttsf_mae"] += F.l1_loss(y_base, y, reduction="sum").item() / (
                    y.size(1) * y.size(2)
                )

    result = {key: value / max(count, 1) for key, value in sums.items()}
    if latenttsf_baseline is None:
        result.pop("latenttsf_mse")
        result.pop("latenttsf_mae")
    return result


def save_checkpoint(path, args, model, autoencoder, epoch, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "autoencoder_state_dict": autoencoder.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        path,
    )


def train(args, device):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    autoencoder = load_autoencoder(args, device)
    branch_oracle, branch_args = load_branch_oracle(args, device)
    latenttsf_baseline = load_latenttsf_baseline(args, device)

    model = BranchRegularizedForecaster(args, branch_args.num_branches).to(device)
    if args.init_from_latenttsf:
        state = load_latenttsf_state(args, device)
        if state is None:
            raise ValueError("--init_from_latenttsf requires --latenttsf_checkpoint")
        model.forecaster.load_state_dict(state)
        print(f"Initialized G from LatentTSF checkpoint: {args.latenttsf_checkpoint}", flush=True)
    model_main_params = list(model.main_parameters())
    main_clip_params = list(model_main_params)
    main_param_groups = [{"params": model_main_params, "lr": args.lr}]
    if args.unfreeze_encoder:
        enc = autoencoder.module.encoder if hasattr(autoencoder, "module") else autoencoder.encoder
        encoder_params = list(enc.parameters())
        main_clip_params.extend(encoder_params)
        main_param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(main_param_groups, weight_decay=args.weight_decay)
    branch_optimizer = torch.optim.AdamW(model.branch_parameters(), lr=args.branch_lr, weight_decay=args.weight_decay)

    train_loader = official_loader(args, "train", shuffle=True)
    val_loader = official_loader(args, "val", shuffle=False)
    test_loader = official_loader(args, "test", shuffle=False)

    print(
        "Branch-regularized latent forecaster | "
        f"probe-only branch tau={args.branch_tau} "
        f"lambda_latent={args.lambda_latent} lambda_delta={args.lambda_delta} "
        f"unfreeze_encoder={args.unfreeze_encoder}",
        flush=True,
    )
    print("Branch head uses stopgrad(z') and never updates E/A/G; test prediction is y_hat = decode(G(A(E(x)))).", flush=True)

    best_path = os.path.join(args.output_dir, "best_branch_regularized_latent_forecaster.pt")
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "total": 0.0,
            "forecast_mse": 0.0,
            "forecast_mae": 0.0,
            "latent_mse": 0.0,
            "delta_mse": 0.0,
            "branch_kl": 0.0,
            "branch_winner_at_1": 0.0,
            "branch_winner_at_3": 0.0,
            "oracle_best_mse": 0.0,
        }
        count = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            total, branch_loss, metrics = compute_losses(args, model, autoencoder, branch_oracle, batch_x, batch_y, device)
            optimizer.zero_grad()
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(main_clip_params, args.grad_clip)
            optimizer.step()

            branch_optimizer.zero_grad()
            (args.lambda_branch * branch_loss).backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.branch_parameters(), args.grad_clip)
            branch_optimizer.step()

            bsz = batch_x.size(0)
            count += bsz
            for key in sums:
                sums[key] += metrics[key].item() * bsz

        train_metrics = {key: value / max(count, 1) for key, value in sums.items()}
        val_metrics = evaluate(args, model, autoencoder, branch_oracle, val_loader, device, latenttsf_baseline)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"epoch {epoch:03d} | "
            f"train forecast {train_metrics['forecast_mse']:.6f}/{train_metrics['forecast_mae']:.6f} "
            f"branchKL {train_metrics['branch_kl']:.4f} w@1 {train_metrics['branch_winner_at_1']:.3f} | "
            f"val forecast {val_metrics['forecast_mse']:.6f}/{val_metrics['forecast_mae']:.6f} "
            f"branchKL {val_metrics['branch_kl']:.4f} w@1 {val_metrics['branch_winner_at_1']:.3f}",
            flush=True,
        )

        metric = val_metrics[args.early_stop_metric]
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, args, model, autoencoder, epoch, val_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.unfreeze_encoder and "autoencoder_state_dict" in checkpoint:
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    val_metrics = evaluate(args, model, autoencoder, branch_oracle, val_loader, device, latenttsf_baseline)
    test_metrics = evaluate(args, model, autoencoder, branch_oracle, test_loader, device, latenttsf_baseline)

    summary = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "branch_checkpoint": args.branch_checkpoint,
        "latenttsf_checkpoint": args.latenttsf_checkpoint,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(args.output_dir, "branch_regularized_latent_forecaster_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBranch-regularized latent forecaster [test]")
    print(f"  forecast MSE/MAE:    {test_metrics['forecast_mse']:.6f} / {test_metrics['forecast_mae']:.6f}")
    if latenttsf_baseline is not None:
        print(f"  LatentTSF MSE/MAE:   {test_metrics['latenttsf_mse']:.6f} / {test_metrics['latenttsf_mae']:.6f}")
    print(f"  oracle branch MSE:   {test_metrics['oracle_best_mse']:.6f}")
    print(f"  branch KL:           {test_metrics['branch_kl']:.6f}")
    print(f"  branch winner@1/@3:  {test_metrics['branch_winner_at_1']:.4f} / {test_metrics['branch_winner_at_3']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Latent forecaster with train-only branch mode regularization")
    parser.add_argument("--output_dir", type=str, default="./latent_outputs/branch_regularized_latent_forecaster")
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--branch_checkpoint", type=str, required=True)
    parser.add_argument("--latenttsf_checkpoint", type=str, default="")
    parser.add_argument("--init_from_latenttsf", action="store_true", default=False)

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
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--adapter_hidden_dim", type=int, default=64)
    parser.add_argument("--branch_hidden_dim", type=int, default=64)
    parser.add_argument("--branch_tau", type=float, default=0.1)
    parser.add_argument("--lambda_branch", type=float, default=0.1)
    parser.add_argument("--lambda_latent", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.1)
    parser.add_argument("--unfreeze_encoder", action="store_true", default=False)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--branch_lr", type=float, default=1e-3)

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
        default="forecast_mse",
        choices=["total", "forecast_mse", "latent_mse", "branch_kl"],
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
