import argparse
import json
import os
import subprocess
import sys


def default_checkpoint(root, k):
    return os.path.join(root, f"official_multibranch_ETTh1_sl96_pl96_normK{k}", "best_multibranch_official.pt")


def read_metrics(out_dir):
    path = os.path.join(out_dir, "branch_probe_metrics.json")
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run branch probe K sweep over existing branch checkpoints")
    parser.add_argument("--latent_root", type=str, default="./latent_outputs")
    parser.add_argument("--output_root", type=str, default="./latent_outputs/branch_probe_k_sweep")
    parser.add_argument("--ks", type=str, default="4,8,16,32")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    summary = {}
    for k in [int(item.strip()) for item in args.ks.split(",") if item.strip()]:
        ckpt = default_checkpoint(args.latent_root, k)
        out_dir = os.path.join(args.output_root, f"K{k}")
        if not os.path.exists(ckpt):
            summary[str(k)] = {
                "status": "missing",
                "branch_checkpoint": ckpt,
                "message": "No branch bank checkpoint found. Generate this K first.",
            }
            print(f"K={k}: missing checkpoint {ckpt}", flush=True)
            continue

        cmd = [
            args.python,
            "-u",
            "train_branch_probe.py",
            "--output_dir",
            out_dir,
            "--autoencoder_path",
            args.autoencoder_path,
            "--branch_checkpoint",
            ckpt,
            "--model",
            "DLinear",
            "--task_name",
            "long_term_forecast",
            "--data",
            "ETTh1",
            "--root_path",
            "./dataset/ETT-small/",
            "--data_path",
            "ETTh1.csv",
            "--features",
            "M",
            "--target",
            "OT",
            "--freq",
            "h",
            "--seq_len",
            "96",
            "--label_len",
            "0",
            "--pred_len",
            "96",
            "--step",
            "1",
            "--enc_in",
            "7",
            "--dec_in",
            "7",
            "--c_out",
            "7",
            "--d_model",
            "32",
            "--d_ff",
            "64",
            "--ae_type",
            "MLP",
            "--moving_avg",
            "25",
            "--num_branches",
            str(k),
            "--hidden_dim",
            "512",
            "--probe_hidden_dim",
            "128",
            "--branch_tau",
            "0.1",
            "--lambda_quality",
            "1.0",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
        print(f"K={k}: running probe from {ckpt}", flush=True)
        subprocess.run(cmd, check=True)
        metrics = read_metrics(out_dir)
        summary[str(k)] = {
            "status": "ok",
            "branch_checkpoint": ckpt,
            "oracle_best_mse": metrics["test"]["oracle_best_mse"],
            "branch_diversity": metrics["test"]["branch_diversity"],
            "winner_at_1": metrics["test"]["winner_at_1"],
            "winner_at_3": metrics["test"]["winner_at_3"],
            "mode_kl": metrics["test"]["mode_kl"],
            "quality_corr": metrics["test"]["quality_corr"],
            "winner_entropy": metrics["test"]["winner_entropy"],
            "winner_top5_coverage": metrics["test"]["winner_top5_coverage"],
            "drift": metrics["drift"],
        }

    out_path = os.path.join(args.output_root, "k_sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved K sweep summary: {out_path}", flush=True)


if __name__ == "__main__":
    main()
