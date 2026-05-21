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
from models.iTransformer import Model as ITransformer


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.dataset[index]
        return index, batch_x, batch_y, batch_x_mark, batch_y_mark


@dataclass
class FuturePriorMemory:
    keys: torch.Tensor
    past: torch.Tensor
    future: torch.Tensor
    past_mu: torch.Tensor
    past_std: torch.Tensor


class RAFTITransformer(nn.Module):
    def __init__(self, args, with_prior):
        super().__init__()
        input_dim = args.enc_in * 2 if with_prior else args.enc_in
        cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            d_model=args.d_model,
            embed=args.embed,
            freq=args.freq,
            dropout=args.dropout,
            factor=args.factor,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            activation=args.activation,
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=input_dim,
        )
        self.backbone = ITransformer(cfg)
        self.out_dim = args.enc_in

    def forward(self, x_aug):
        out = self.backbone(x_aug, None, None, None)
        return out[:, :, : self.out_dim]


def parse_args():
    parser = argparse.ArgumentParser(description="RAFT-style candidate future prior experiment on ETTh1.")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enc_in", type=int, default=7)

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--self_mask_window", type=int, default=192)
    parser.add_argument("--key_mode", type=str, default="flatten", choices=["flatten", "mean"])
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["pure_no_context", "random_future_prior", "shuffled_future_prior", "retrieved_future_prior"],
        choices=["pure_no_context", "random_future_prior", "shuffled_future_prior", "retrieved_future_prior"],
    )

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--activation", type=str, default="gelu")

    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", type=str, default="./results/raft_future_prior/ETTh1_obs_itransformer")
    parser.add_argument("--smoke", action="store_true", default=False)
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


def data_args(args):
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
        embed=args.embed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_ratio=0,
    )


def make_loader(args, flag, shuffle, indexed):
    dataset, _ = data_provider(data_args(args), flag)
    if indexed:
        dataset = IndexedDataset(dataset)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=False)


def instance_stats(x):
    mu = x.mean(dim=1, keepdim=True)
    std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return mu, std


def normalized_window(x):
    mu, std = instance_stats(x)
    return (x - mu) / std


def make_key(args, x):
    x_norm = normalized_window(x)
    if args.key_mode == "mean":
        return x_norm.mean(dim=1)
    if args.key_mode == "flatten":
        return x_norm.flatten(start_dim=1)
    raise ValueError(f"Unknown key_mode: {args.key_mode}")


def build_memory(args, device):
    loader = make_loader(args, "train", shuffle=False, indexed=True)
    keys, past, future, mus, stds = [], [], [], [], []
    for _, batch_x, batch_y, _, _ in tqdm(loader, desc="Building RAFT future-prior memory"):
        x = batch_x.float()
        y = batch_y[:, -args.pred_len :, :].float()
        mu, std = instance_stats(x)
        keys.append(make_key(args, x))
        past.append(x)
        future.append(y)
        mus.append(mu)
        stds.append(std)
    memory = FuturePriorMemory(
        keys=F.normalize(torch.cat(keys, dim=0), dim=-1).to(device),
        past=torch.cat(past, dim=0).to(device),
        future=torch.cat(future, dim=0).to(device),
        past_mu=torch.cat(mus, dim=0).to(device),
        past_std=torch.cat(stds, dim=0).to(device),
    )
    print(
        f"Memory verified: keys={tuple(memory.keys.shape)}, past={tuple(memory.past.shape)}, "
        f"future={tuple(memory.future.shape)}"
    )
    return memory


def exclusion_mask(args, bsz, memory_size, sample_indices, device):
    if sample_indices is None:
        return torch.zeros(bsz, memory_size, dtype=torch.bool, device=device)
    positions = torch.arange(memory_size, device=device).view(1, -1)
    queries = sample_indices.to(device).view(-1, 1)
    return (positions - queries).abs() <= args.self_mask_window


def align_future(y_i, mu_i, std_i, mu_q, std_q):
    y_norm = (y_i - mu_i) / (std_i + 1e-5)
    return y_norm * std_q + mu_q


def retrieve_prior(args, x_q, memory, group, sample_indices=None, generator=None):
    bsz = x_q.shape[0]
    device = x_q.device
    mu_q, std_q = instance_stats(x_q)

    if group == "pure_no_context":
        return torch.zeros(bsz, args.pred_len, args.enc_in, device=device)

    if group == "random_future_prior":
        valid = (~exclusion_mask(args, bsz, memory.future.shape[0], sample_indices, device)).float()
        if torch.any(valid.sum(dim=1) < args.k):
            raise RuntimeError("Not enough valid candidates after self-mask.")
        idx = torch.multinomial(valid, args.k, replacement=False, generator=generator)
        weights = torch.full((bsz, args.k), 1.0 / args.k, device=device)
    else:
        q = F.normalize(make_key(args, x_q), dim=-1)
        sim = q @ memory.keys.T
        sim = sim.masked_fill(exclusion_mask(args, bsz, memory.future.shape[0], sample_indices, device), -torch.inf)
        sim_topk, idx = torch.topk(sim, k=args.k, dim=1)
        weights = torch.softmax(sim_topk / args.tau, dim=1)

    y_topk = memory.future[idx]
    mu_topk = memory.past_mu[idx]
    std_topk = memory.past_std[idx]
    aligned = align_future(y_topk, mu_topk, std_topk, mu_q.unsqueeze(1), std_q.unsqueeze(1))
    prior = (aligned * weights[:, :, None, None]).sum(dim=1)

    if group == "shuffled_future_prior":
        if bsz > 1:
            perm = torch.randperm(bsz, device=device, generator=generator)
            prior = prior[perm]
        else:
            prior = torch.zeros_like(prior)
    elif group not in {"random_future_prior", "retrieved_future_prior"}:
        raise ValueError(f"Unknown group: {group}")
    return prior


def build_model_input(args, x_q, y_prior, group):
    if group == "pure_no_context":
        return x_q
    if y_prior.shape[1] == args.seq_len:
        prior_context = y_prior
    else:
        prior_context = F.interpolate(
            y_prior.transpose(1, 2),
            size=args.seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return torch.cat([x_q, prior_context], dim=-1)


def cosine_copy_ratio(y_hat, y_prior):
    if torch.all(y_prior == 0):
        return torch.zeros((), device=y_hat.device)
    return F.cosine_similarity(y_hat.flatten(start_dim=1), y_prior.flatten(start_dim=1), dim=-1).mean()


def empty_stats():
    return {
        "sqerr": 0.0,
        "abserr": 0.0,
        "count": 0,
        "loss_sum": 0.0,
        "prior_sqerr": 0.0,
        "copy_ratio_sum": 0.0,
        "prior_norm_sum": 0.0,
        "true_norm_sum": 0.0,
        "batch_count": 0,
    }


def update_stats(stats, loss, y_hat, y_true, y_prior):
    diff = y_hat - y_true
    prior_diff = y_prior - y_true
    stats["sqerr"] += float((diff ** 2).sum().item())
    stats["abserr"] += float(diff.abs().sum().item())
    stats["count"] += int(diff.numel())
    stats["loss_sum"] += float(loss.item())
    stats["prior_sqerr"] += float((prior_diff ** 2).sum().item())
    stats["copy_ratio_sum"] += float(cosine_copy_ratio(y_hat, y_prior).item())
    stats["prior_norm_sum"] += float(y_prior.norm(dim=-1).mean().item())
    stats["true_norm_sum"] += float(y_true.norm(dim=-1).mean().item())
    stats["batch_count"] += 1


def finalize_stats(stats):
    batches = max(stats["batch_count"], 1)
    return {
        "mse": stats["sqerr"] / max(stats["count"], 1),
        "mae": stats["abserr"] / max(stats["count"], 1),
        "loss": stats["loss_sum"] / batches,
        "prior_mse": stats["prior_sqerr"] / max(stats["count"], 1),
        "copy_ratio": stats["copy_ratio_sum"] / batches,
        "prior_norm": stats["prior_norm_sum"] / batches,
        "true_norm": stats["true_norm_sum"] / batches,
    }


def run_epoch(args, model, memory, loader, group, device, optimizer=None, generator=None):
    training = optimizer is not None
    model.train(training)
    stats = empty_stats()
    is_indexed = isinstance(loader.dataset, IndexedDataset)

    for batch in tqdm(loader, desc=f"{'Train' if training else 'Eval'} {group}"):
        if is_indexed:
            sample_idx, batch_x, batch_y, _, _ = batch
            sample_idx = sample_idx.to(device)
        else:
            sample_idx = None
            batch_x, batch_y, _, _ = batch
        x_q = batch_x.float().to(device)
        y_true = batch_y[:, -args.pred_len :, :].float().to(device)

        with torch.no_grad():
            y_prior = retrieve_prior(args, x_q, memory, group, sample_idx, generator)
            x_aug = build_model_input(args, x_q, y_prior, group)

        y_hat = model(x_aug)
        loss = F.mse_loss(y_hat, y_true)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        update_stats(stats, loss.detach(), y_hat.detach(), y_true, y_prior)

        if args.smoke and stats["batch_count"] >= 3:
            break
    return finalize_stats(stats)


def verify_model(args, model, group, device):
    input_dim = args.enc_in if group == "pure_no_context" else args.enc_in * 2
    with torch.no_grad():
        x = torch.zeros(2, args.seq_len, input_dim, device=device)
        y = model(x)
    assert y.shape == (2, args.pred_len, args.enc_in), tuple(y.shape)
    print(f"Pipeline verified: group={group}, input={tuple(x.shape)}, pred={tuple(y.shape)}")


def train_group(args, memory, group, device):
    set_seed(args.seed)
    model = RAFTITransformer(args, with_prior=(group != "pure_no_context")).to(device)
    verify_model(args, model, group, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(args, "train", shuffle=True, indexed=True)
    val_loader = make_loader(args, "val", shuffle=False, indexed=False)
    test_loader = make_loader(args, "test", shuffle=False, indexed=False)
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(args.seed + 17)

    history = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.train_epochs + 1):
        train_metrics = run_epoch(args, model, memory, train_loader, group, device, optimizer, train_generator)
        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(args.seed + 1000 + epoch)
        val_metrics = run_epoch(args, model, memory, val_loader, group, device, None, val_generator)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"{group} epoch {epoch}: train={train_metrics} val={val_metrics}")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping {group} at epoch {epoch}; best_val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_generator = torch.Generator(device=device)
    test_generator.manual_seed(args.seed + 101)
    test_metrics = run_epoch(args, model, memory, test_loader, group, device, None, test_generator)
    return history, test_metrics


def write_csv(path, rows):
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.smoke:
        args.train_epochs = min(args.train_epochs, 1)
        args.output_dir = os.path.join(args.output_dir, "smoke")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Config: K={args.k}, tau={args.tau}, key_mode={args.key_mode}, self_mask_window={args.self_mask_window}")

    memory = build_memory(args, device)
    results, rows = {}, []
    for group in args.groups:
        print(f"Running {group}")
        history, test_metrics = train_group(args, memory, group, device)
        results[group] = {"group": group, "history": history, "test": test_metrics}
        rows.append({"group": group, **test_metrics})

    write_csv(os.path.join(args.output_dir, "metrics.csv"), rows)
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"Saved RAFT future-prior results to: {args.output_dir}")


if __name__ == "__main__":
    main()
