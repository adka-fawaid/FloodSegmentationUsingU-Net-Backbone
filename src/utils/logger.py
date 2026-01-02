import csv
import os

def init_csv_log(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

def append_csv_log(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
def append_summary(summary_path, row_dict):
    """
    row_dict contoh:
    {
        "model": "unet",
        "batch": 4,
        "lr": 0.001,
        "optimizer": "AdamW",
        "epochs": 40,
        "final_val_iou": 0.8123,
        "final_test_iou": 0.7981,
        "weights_path": "saved_models/best_unet.pth",
        "results_path": "results/unet/predictions/"
    }
    """
    header = list(row_dict.keys())
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([row_dict[k] for k in header])
