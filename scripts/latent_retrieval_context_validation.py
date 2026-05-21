import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.data_factory import data_provider
from my_AE import get_autoencoder
from my_utils import model_dict


DEFAULT_ETTH1_AE = (
    "./checkpoints/"
    "AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0/"
    "checkpoint.pth"
)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.dataset[index]
        return index, batch_x, batch_y, batch_x_mark, batch_y_mark


class IdentityFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, d_model)
        self.reset_identity()

    def reset_identity(self):
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[:, : self.linear.out_features].copy_(torch.eye(self.linear.out_features))
            self.linear.bias.zero_()

    def forward(self, z_q, context):
        if context.shape[1] != z_q.shape[1]:
            context = F.interpolate(
                context.transpose(1, 2),
                size=z_q.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        z_cat = torch.cat([z_q, context], dim=-1)
        return self.linear(z_cat)

    def context_weight_norm(self):
        d_model = self.linear.out_features
        return self.linear.weight[:, d_model:].norm().item()


def build_dec_inp_latent(z_x, label_len, pred_len, d_model, device):
    batch_size = z_x.size(0)
    dec_inp_pred = torch.zeros(batch_size, pred_len, d_model).float().to(device)
    if label_len > 0:
        dec_inp_label = z_x[:, -label_len:, :]
        return torch.cat([dec_inp_label, dec_inp_pred], dim=1)
    return dec_inp_pred


@dataclass
class MemoryBank:
    keys: torch.Tensor
    values: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict frozen-forecaster latent retrieval-as-context validation."
    )
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--model", type=str, default="DLinear")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)
    parser.add_argument("--ae_type", type=str, default="MLP")
    parser.add_argument("--autoencoder_path", type=str, default=DEFAULT_ETTH1_AE)
    parser.add_argument("--forecaster_path", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--anchor_loss_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", type=str, default="./results/latent_retrieval_context_validation/ETTh1")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name):
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if name == "mps" or (name == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


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


def build_forecaster_args(args):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        enc_in=args.d_model,
        dec_in=args.d_model,
        c_out=args.d_model,
        d_model=args.d_model,
        d_ff=args.d_ff,
        moving_avg=args.moving_avg,
        individual=args.individual,
        embed="timeF",
        freq=args.freq,
        dropout=0.0,
        factor=1,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        activation="gelu",
        output_attention=False,
        distil=True,
        top_k=5,
        num_kernels=6,
    )


def load_state_dict(path, device):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_models(args, device):
    ae = get_autoencoder(build_ae_args(args)).float().to(device)
    ae.load_state_dict(load_state_dict(args.autoencoder_path, device))
    ae.eval()

    if args.model not in model_dict:
        raise ValueError(f"Unknown model {args.model}. Available: {sorted(model_dict)}")
    forecaster = model_dict[args.model].Model(build_forecaster_args(args)).float().to(device)
    forecaster.load_state_dict(load_state_dict(args.forecaster_path, device))
    forecaster.eval()

    for model in (ae, forecaster):
        for param in model.parameters():
            param.requires_grad = False
    return ae, forecaster


def make_loader(args, flag, shuffle, indexed):
    dataset, _ = data_provider(build_data_args(args), flag)
    if indexed:
        dataset = IndexedDataset(dataset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )


def build_memory_bank(args, ae, device):
    loader = make_loader(args, "train", shuffle=False, indexed=True)
    keys, values = [], []
    with torch.no_grad():
        for _, batch_x, batch_y, _, _ in tqdm(loader, desc="Building memory bank"):
            batch_x = batch_x.float().to(device)
            future_y = batch_y[:, -args.pred_len :, :].float().to(device)
            z_x = ae.encode(batch_x)
            z_y = ae.encode(future_y)
            key = z_x.mean(dim=1)
            anchor = z_x[:, -1:, :]
            value = z_y - anchor
            keys.append(key.cpu())
            values.append(value.cpu())
    return MemoryBank(keys=torch.cat(keys, dim=0).to(device), values=torch.cat(values, dim=0).to(device))


def retrieve_context(args, z_q, memory, mode, sample_indices=None, generator=None):
    bsz = z_q.shape[0]
    q = F.normalize(z_q.mean(dim=1), dim=-1)
    keys = F.normalize(memory.keys, dim=-1)
    sim = q @ keys.T

    if sample_indices is not None:
        sim[torch.arange(bsz, device=sim.device), sample_indices.to(sim.device)] = -torch.inf

    if mode == "no_context":
        return torch.zeros(bsz, memory.values.shape[1], memory.values.shape[2], device=z_q.device)

    if mode == "random_context":
        random_idx = torch.randint(
            low=0,
            high=memory.values.shape[0],
            size=(bsz, args.topk),
            device=z_q.device,
            generator=generator,
        )
        context = memory.values[random_idx].mean(dim=1)
        return context

    _, top_idx = torch.topk(sim, k=args.topk, dim=1)
    sim_topk = torch.gather(sim, 1, top_idx)
    weights = torch.softmax(sim_topk / args.tau, dim=1)
    context = (memory.values[top_idx] * weights[:, :, None, None]).sum(dim=1)

    if mode == "shuffled_context":
        if bsz > 1:
            perm = torch.randperm(bsz, device=z_q.device, generator=generator)
            context = context[perm]
        else:
            context = torch.zeros_like(context)
    elif mode != "retrieved_context":
        raise ValueError(f"Unknown context mode: {mode}")
    return context


def run_forecaster(args, forecaster, z_aug, batch_x_mark, batch_y_mark, device):
    dec_inp = build_dec_inp_latent(z_aug, args.label_len, args.pred_len, args.d_model, device)
    return forecaster(z_aug, batch_x_mark, dec_inp, batch_y_mark)[:, -args.pred_len :, :]


def batch_target(args, batch_y):
    return batch_y[:, -args.pred_len :, :]


def metric_update(stats, y_hat, y_true, z_aug=None, z_q=None, context=None, fusion=None):
    diff = y_hat - y_true
    stats["sqerr"] += float((diff ** 2).sum().item())
    stats["abserr"] += float(diff.abs().sum().item())
    stats["count"] += int(diff.numel())

    if z_aug is not None:
        delta = z_aug - z_q
        stats["z_abs_delta_sum"] += float(delta.abs().sum().item())
        stats["z_delta_sq_sum"] += float((delta ** 2).sum().item())
        stats["z_count"] += int(delta.numel())
        stats["z_aug_sum"] += float(z_aug.sum().item())
        stats["z_aug_sq_sum"] += float((z_aug ** 2).sum().item())
        stats["z_q_sum"] += float(z_q.sum().item())
        stats["z_q_sq_sum"] += float((z_q ** 2).sum().item())
        if context is not None:
            stats["context_abs_sum"] += float(context.abs().sum().item())
            stats["context_count"] += int(context.numel())


def empty_stats():
    return {
        "sqerr": 0.0,
        "abserr": 0.0,
        "count": 0,
        "z_abs_delta_sum": 0.0,
        "z_delta_sq_sum": 0.0,
        "z_count": 0,
        "z_aug_sum": 0.0,
        "z_aug_sq_sum": 0.0,
        "z_q_sum": 0.0,
        "z_q_sq_sum": 0.0,
        "context_abs_sum": 0.0,
        "context_count": 0,
    }


def finalize_stats(stats, fusion=None):
    out = {
        "mse": stats["sqerr"] / max(stats["count"], 1),
        "mae": stats["abserr"] / max(stats["count"], 1),
    }
    if stats["z_count"] > 0:
        z_aug_mean = stats["z_aug_sum"] / stats["z_count"]
        z_q_mean = stats["z_q_sum"] / stats["z_count"]
        z_aug_var = max(stats["z_aug_sq_sum"] / stats["z_count"] - z_aug_mean ** 2, 0.0)
        z_q_var = max(stats["z_q_sq_sum"] / stats["z_count"] - z_q_mean ** 2, 0.0)
        out.update(
            {
                "mean_abs_z_aug_minus_z_q": stats["z_abs_delta_sum"] / stats["z_count"],
                "mse_z_aug_z_q": stats["z_delta_sq_sum"] / stats["z_count"],
                "std_z_aug_over_std_z_q": (z_aug_var ** 0.5) / max(z_q_var ** 0.5, 1e-12),
                "mean_abs_context": stats["context_abs_sum"] / max(stats["context_count"], 1),
            }
        )
    if fusion is not None:
        out["fusion_context_weight_norm"] = fusion.context_weight_norm()
    return out


def evaluate_original(args, ae, forecaster, loader, device):
    stats = empty_stats()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(loader, desc="Eval original frozen"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            z_q = ae.encode(batch_x)
            z_pred = run_forecaster(args, forecaster, z_q, batch_x_mark, batch_y_mark, device)
            y_hat = ae.decode(z_pred)
            metric_update(stats, y_hat, batch_target(args, batch_y))
    return finalize_stats(stats)


def evaluate_group(args, ae, forecaster, fusion, memory, loader, mode, device, generator):
    fusion.eval()
    stats = empty_stats()
    is_train_loader = isinstance(loader.dataset, IndexedDataset)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {mode}"):
            if is_train_loader:
                sample_idx, batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                sample_idx = sample_idx.to(device)
            else:
                sample_idx = None
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            z_q = ae.encode(batch_x)
            context = retrieve_context(args, z_q, memory, mode, sample_idx, generator)
            z_aug = fusion(z_q, context)
            z_pred = run_forecaster(args, forecaster, z_aug, batch_x_mark, batch_y_mark, device)
            y_hat = ae.decode(z_pred)
            metric_update(stats, y_hat, batch_target(args, batch_y), z_aug, z_q, context)
    return finalize_stats(stats, fusion)


def train_group(args, ae, forecaster, memory, mode, device):
    set_seed(args.seed)
    fusion = IdentityFusion(args.d_model).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate)
    train_loader = make_loader(args, "train", shuffle=True, indexed=True)
    random_gen = torch.Generator(device=device)
    random_gen.manual_seed(args.seed + 17)

    history = []
    for epoch in range(1, args.train_epochs + 1):
        fusion.train()
        losses = []
        for sample_idx, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            train_loader, desc=f"Train {mode} epoch {epoch}/{args.train_epochs}"
        ):
            sample_idx = sample_idx.to(device)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            with torch.no_grad():
                z_q = ae.encode(batch_x)
                context = retrieve_context(args, z_q, memory, mode, sample_idx, random_gen)

            z_aug = fusion(z_q, context)
            z_pred = run_forecaster(args, forecaster, z_aug, batch_x_mark, batch_y_mark, device)
            y_hat = ae.decode(z_pred)
            loss = F.mse_loss(y_hat, batch_target(args, batch_y))
            if args.anchor_loss_weight > 0:
                loss = loss + args.anchor_loss_weight * F.mse_loss(z_aug, z_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    return fusion, history


def write_summary_csv(path, results):
    rows = []
    for group, metrics in results.items():
        row = {"group": group}
        row.update(metrics)
        rows.append(row)
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    ae, forecaster = load_models(args, device)
    memory = build_memory_bank(args, ae, device)
    print(f"Memory keys: {tuple(memory.keys.shape)}, values: {tuple(memory.values.shape)}")

    test_loader = make_loader(args, "test", shuffle=False, indexed=False)
    results = {"original_frozen": evaluate_original(args, ae, forecaster, test_loader, device)}
    histories = {}

    for mode in ["no_context", "random_context", "shuffled_context", "retrieved_context"]:
        fusion, history = train_group(args, ae, forecaster, memory, mode, device)
        histories[mode] = history
        group_gen = torch.Generator(device=device)
        group_gen.manual_seed(args.seed + 101)
        results[mode] = evaluate_group(args, ae, forecaster, fusion, memory, test_loader, mode, device, group_gen)
        torch.save(fusion.state_dict(), os.path.join(args.output_dir, f"fusion_{mode}.pth"))

    payload = {"config": vars(args), "results": results, "history": histories}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_summary_csv(os.path.join(args.output_dir, "metrics.csv"), results)
    print(json.dumps(payload, indent=2))
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
