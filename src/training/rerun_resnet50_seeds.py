import os
import yaml
import csv
import subprocess
import argparse

# Fungsi untuk menjalankan training ulang dengan seed berbeda

def run_repeats(config_path, seeds, output_dir):
    # Pastikan path absolut
    config_path = os.path.abspath(config_path)
    output_dir = os.path.abspath(output_dir)
    train_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "train.py"))
    
    # Buat subfolder berdasarkan nama config (tanpa .yaml extension)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    config_output_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)
    
    results = []
    for seed in seeds:
        print(f"\n=== Training ulang dengan seed {seed} ===")
        seed_dir = os.path.join(config_output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        cmd = f'python "{train_py}" --config "{config_path}" --seed {seed} --output_dir "{seed_dir}"'
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"Training gagal untuk seed {seed}")
            continue
        # Ambil metrik dari summary CSV (misal Results/metrics_summary.csv di seed_dir)
        summary_path = os.path.join(seed_dir, "metrics_summary.csv")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) > 0:
                    last = rows[-1]
                    # Extract best_val_iou and best_epoch for val metrics
                    best_val_iou = last.get("best_val_iou", "-")
                    best_epoch = last.get("best_epoch", None)
                    val_iou = best_val_iou
                    val_dice = "-"
                    # Try to get val_dice from training_log.csv if possible
                    log_path = os.path.join(seed_dir, "training_log.csv")
                    if best_epoch is not None and os.path.exists(log_path):
                        try:
                            with open(log_path) as flog:
                                log_reader = csv.DictReader(flog)
                                for row in log_reader:
                                    if str(row.get("epoch")) == str(best_epoch):
                                        val_dice = row.get("val_dice", "-")
                                        break
                        except Exception as e:
                            print(f"Gagal membaca val_dice dari training_log.csv: {e}")
                    results.append({
                        "seed": seed,
                        "test_iou": last.get("test_iou", "-"),
                        "test_dice": last.get("test_dice", "-"),
                        "val_iou": val_iou,
                        "val_dice": val_dice,
                        "train_loss": "-"
                    })
        else:
            print(f"metrics_summary.csv tidak ditemukan untuk seed {seed}")
    # Ambil parameter dari best config
    with open(config_path) as f:
        best_cfg = yaml.safe_load(f)
    param_fields = [
        "model", "encoder", "base_c", "optimizer", "batch_size", "lr", "epochs", "loss_type", "focal_alpha", "focal_gamma", "bce_dice_weight", "use_scheduler", "scheduler_type", "warmup_epochs", "accum_steps", "use_amp"
    ]
    config_source = os.path.basename(config_path)
    param_row = {k: best_cfg.get(k, '-') for k in param_fields}
    param_row['config_source'] = config_source
    # Simpan rekap ke CSV (per config) di subfolder config
    out_csv = os.path.join(config_output_dir, f"rerun_seeds_summary.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = param_fields + ["config_source", "seed", "test_iou", "test_dice", "val_iou", "val_dice", "train_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_full = {**param_row, **row}
            writer.writerow(row_full)
    print(f"\n✓ Rekap hasil rerun 3 seed disimpan di: {out_csv}")

def run_efficientnet_random():
    """Run EfficientNet B1 dengan random config (3 seeds)"""
    # Buat random config untuk EfficientNet
    random_config = {
        "model": "unet_efficientnet_b1",
        "encoder": "efficientnet_b1",
        "base_c": 96,
        "optimizer": "adam",
        "lr": 0.001,  # Random config: lr lebih besar
        "batch_size": 4,
        "epochs": 100,
        "loss_type": "focal_dice",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "bce_dice_weight": 0.5,
        "use_scheduler": True,
        "scheduler_type": "cosine",
        "warmup_epochs": 5,
        "accum_steps": 1,
        "use_amp": True,
        "split_json": "Data/splits/splits.json",
        "checkpoint_dir": "Save_models",
        "save_name": "random_unet_efficientnet_b1.pth",
        "metrics_out": "Results/unet_efficientnet_b1/random_metrics_summary.csv"
    }
    
    # Simpan random config
    config_dir = os.path.abspath("Results/OFAT_efficientnet")
    os.makedirs(config_dir, exist_ok=True)
    random_config_path = os.path.join(config_dir, "random_config.yaml")
    
    with open(random_config_path, "w") as f:
        yaml.dump(random_config, f, default_flow_style=False)
    
    print(f"✓ Random config untuk EfficientNet dibuat: {random_config_path}")
    
    # Run dengan 3 seeds
    output_dir = os.path.abspath("Results/EfficientNet_rerun_seeds")
    seeds = [0, 42, 123]
    run_repeats(random_config_path, seeds, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun training 3x dengan seed berbeda")
    parser.add_argument("--config", default="../../Results/OFAT_resnet50/best_config.yaml", help="Path ke best config YAML")
    parser.add_argument("--output_dir", default="../../Results/ResNet50_rerun_seeds", help="Folder output rekap rerun")
    parser.add_argument("--run_efficientnet_random", action="store_true", help="Run EfficientNet dengan random config")
    args = parser.parse_args()
    
    if args.run_efficientnet_random:
        run_efficientnet_random()
    else:
        seeds = [0, 42, 123]
        run_repeats(args.config, seeds, args.output_dir)
