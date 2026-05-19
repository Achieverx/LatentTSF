import argparse
import csv
import json
import os


def infer_method(summary):
    use_lora = int(summary.get("use_lora", 0))
    calib_type = summary.get("calib_type", "none")
    dynamic_scale = summary.get("dynamic_scale", "none")

    if use_lora == 0 and calib_type == "none":
        return "Baseline"
    if use_lora == 0 and calib_type == "lowrank":
        return "LowRank"
    if use_lora == 1 and calib_type == "none":
        return "LoRA"
    if use_lora == 1 and calib_type == "lowrank":
        return "LoRA+LowRank"
    if use_lora == 0 and calib_type == "block" and dynamic_scale == "none":
        return "StaticBlock"
    if use_lora == 0 and calib_type == "block" and dynamic_scale == "block":
        return "DynamicBlock"
    if use_lora == 1 and calib_type == "block" and dynamic_scale == "block":
        return "LoRA+DynamicBlock"
    return f"lora{use_lora}_{calib_type}_{dynamic_scale}"


def collect_summaries(root_dir, dir_prefix=None):
    rows = []
    for current_root, _, files in os.walk(root_dir):
        if "metrics_summary.json" not in files:
            continue
        base_name = os.path.basename(current_root)
        if dir_prefix and not base_name.startswith(dir_prefix):
            continue
        summary_path = os.path.join(current_root, "metrics_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if "test_adapted_obs_mse" not in summary:
            continue
        rows.append(
            {
                "model": summary.get("model", ""),
                "pred_len": summary.get("pred_len", ""),
                "method": infer_method(summary),
                "use_lora": summary.get("use_lora", 0),
                "calib_type": summary.get("calib_type", ""),
                "dynamic_scale": summary.get("dynamic_scale", ""),
                "test_base_obs_mse": summary.get("test_base_obs_mse", ""),
                "test_adapted_obs_mse": summary.get("test_adapted_obs_mse", ""),
                "test_obs_gain": summary.get("test_obs_gain", ""),
                "test_base_mae": summary.get("test_base_mae", ""),
                "test_adapted_mae": summary.get("test_adapted_mae", ""),
                "trainable_params": summary.get("trainable_params", ""),
                "relative_z_lora_change": summary.get("test_relative_z_lora_change", ""),
                "relative_z_cal_change": summary.get("test_relative_z_cal_change", ""),
                "summary_path": summary_path,
            }
        )
    rows.sort(key=lambda row: (row["model"], int(row["pred_len"]), row["method"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize LoRA + low-rank latent adaptation results")
    parser.add_argument("--root_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_csv", type=str, default="./results/lora_lowrank_etth1_summary.csv")
    parser.add_argument("--dir_prefix", type=str, default="lora_lowrank_latent_adaptation_")
    args = parser.parse_args()

    rows = collect_summaries(args.root_dir, dir_prefix=args.dir_prefix)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = [
        "model",
        "pred_len",
        "method",
        "use_lora",
        "calib_type",
        "dynamic_scale",
        "test_base_obs_mse",
        "test_adapted_obs_mse",
        "test_obs_gain",
        "test_base_mae",
        "test_adapted_mae",
        "trainable_params",
        "relative_z_lora_change",
        "relative_z_cal_change",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"Collected {len(rows)} result summaries")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
