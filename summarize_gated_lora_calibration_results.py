import argparse
import csv
import json
import os


def infer_method(summary):
    use_lora = int(summary.get("use_lora", 0))
    calib_type = summary.get("calib_type", "none")
    calib_gate = summary.get("calib_gate", summary.get("dynamic_scale", "none"))
    lora_target = summary.get("lora_target", "auto")

    if use_lora == 0 and calib_type == "none":
        return "Baseline"
    if use_lora == 0 and calib_type == "lowrank":
        return "LowRank"
    if use_lora == 1 and calib_type == "none" and lora_target == "qv_only":
        return "qv-LoRA"
    if use_lora == 1 and calib_type == "none":
        return "LoRA"
    if use_lora == 1 and calib_type == "lowrank" and calib_gate == "none":
        return "LoRA+LowRank-old"
    if use_lora == 1 and calib_type == "lowrank" and calib_gate != "none" and lora_target == "qv_only":
        return "qv-Gated-LoRA+LowRank"
    if use_lora == 1 and calib_type == "lowrank" and calib_gate != "none":
        return "Gated-LoRA+LowRank"
    if use_lora == 1 and calib_type == "block" and calib_gate != "none":
        return "Gated-LoRA+DynamicBlock"
    return f"lora{use_lora}_{lora_target}_{calib_type}_{calib_gate}"


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
        rows.append(
            {
                "model": summary.get("model", ""),
                "pred_len": summary.get("pred_len", ""),
                "method": infer_method(summary),
                "lora_target": summary.get("lora_target", ""),
                "use_lora": summary.get("use_lora", 0),
                "calib_type": summary.get("calib_type", ""),
                "calib_gate": summary.get("calib_gate", summary.get("dynamic_scale", "")),
                "test_base_obs_mse": summary.get("test_base_obs_mse", ""),
                "test_adapted_obs_mse": summary.get("test_adapted_obs_mse", ""),
                "test_obs_gain": summary.get("test_obs_gain", ""),
                "test_base_mae": summary.get("test_base_mae", ""),
                "test_adapted_mae": summary.get("test_adapted_mae", ""),
                "trainable_params": summary.get("trainable_params", ""),
                "relative_z_lora_change": summary.get("test_relative_z_lora_change", ""),
                "relative_z_cal_change": summary.get("test_relative_z_cal_change", ""),
                "gate_mean": summary.get("gate_mean", ""),
                "gate_std": summary.get("gate_std", ""),
                "gate_min": summary.get("gate_min", ""),
                "gate_max": summary.get("gate_max", ""),
                "summary_path": summary_path,
            }
        )
    rows.sort(key=lambda row: (row["model"], int(row["pred_len"]), row["method"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize gated LoRA + latent calibration results")
    parser.add_argument("--root_dir", type=str, default="./checkpoints")
    parser.add_argument("--dir_prefix", type=str, default="gated_lora_calibration_")
    parser.add_argument("--output_csv", type=str, default="./results/gated_lora_calibration_etth1_summary.csv")
    args = parser.parse_args()

    rows = collect_summaries(args.root_dir, dir_prefix=args.dir_prefix)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = [
        "model",
        "pred_len",
        "method",
        "lora_target",
        "use_lora",
        "calib_type",
        "calib_gate",
        "test_base_obs_mse",
        "test_adapted_obs_mse",
        "test_obs_gain",
        "test_base_mae",
        "test_adapted_mae",
        "trainable_params",
        "relative_z_lora_change",
        "relative_z_cal_change",
        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
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
