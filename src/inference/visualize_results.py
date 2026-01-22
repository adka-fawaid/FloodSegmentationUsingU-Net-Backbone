

import os, sys, json, argparse, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.Dataset.loader import FloodDataset


def load_model(config, device):
    """Load U-Net model with specified encoder. Returns None if checkpoint not found."""
    base_c = config.get('base_c', 32)
    encoder = config.get('encoder', 'baseline')
    
    model = UNet(in_ch=3, n_classes=1, base_c=base_c, encoder=encoder)
    ckpt = os.path.join(config["checkpoint_dir"], config["save_name"])

    # Check if model file exists
    if not os.path.exists(ckpt):
        print(f"⚠ Model weights not found: {ckpt} (skipping this model)")
        return None
    
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    return model.to(device).eval()


def get_best_epoch(config, device):
    """Extract best epoch dari checkpoint metadata"""
    ckpt_path = os.path.join(config["checkpoint_dir"], config["save_name"])
    
    try:
        # Coba load full checkpoint (yang punya metadata)
        ckpt_full = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt_full, dict) and 'epoch' in ckpt_full:
            return ckpt_full['epoch'] + 1  # epoch di-save 0-indexed, jadi +1
    except:
        pass
    
    # Fallback: ambil dari config
    return config.get('epochs', 'unknown')


def arr_mask(arr):
    return (arr > 0.5).astype(np.uint8)


def make_error_map(gt, pred):
    # gt, pred: uint8 0/1 masks
    # Red = False Negative (missed flood), Green = False Positive (false alarm)
    err = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    FN = (gt == 1) & (pred == 0)
    FP = (gt == 0) & (pred == 1)
    err[FN] = [255, 0, 0]  # Red: missed flood
    err[FP] = [0, 255, 0]  # Green: false alarm
    return Image.fromarray(err)


def grid_4(orig, gt, pred, err, out_path, labels):
    """Create horizontal 4-panel layout: Original | Mask | Prediction | Error"""
    orig = orig.resize((256,256))
    gt = gt.resize((256,256))
    pred = pred.resize((256,256))
    err = err.resize((256,256))

    w, h = 256, 256
    spacing = 10  # Space between panels
    label_height = 30  # Tinggi area untuk label di atas
    
    # Canvas horizontal: 4 panels + 3 spacings + ruang label di atas
    canvas = Image.new("RGB", (w*4 + spacing*3, h + label_height), color=(255,255,255))
    
    # Paste panels dengan spacing, mulai dari y=label_height (bawah label)
    canvas.paste(orig, (0, label_height))
    canvas.paste(gt.convert('RGB'), (w + spacing, label_height))
    canvas.paste(pred.convert('RGB'), ((w + spacing)*2, label_height))
    canvas.paste(err, ((w + spacing)*3, label_height))

    draw = ImageDraw.Draw(canvas)
    try:
        fnt = ImageFont.truetype("arial.ttf", 16)
    except:
        fnt = None
    
    # Tambahkan label di atas setiap panel
    for i, label in enumerate(labels):
        x_pos = i * (w + spacing) + w//2  # Center horizontal
        # Teks di tengah area label
        draw.text((x_pos, 8), label, fill=(0,0,0), font=fnt, anchor="mt")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def compute_iou(gt, pred):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    inter = ((gt == 1) & (pred == 1)).sum()
    union = ((gt == 1) | (pred == 1)).sum()
    return float(inter/union) if union>0 else 0.0


def compute_dice(gt, pred):
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    inter = ((gt == 1) & (pred == 1)).sum()
    denominator = gt.sum() + pred.sum()
    return float(2*inter/denominator) if denominator>0 else 0.0


def main(config_baseline_path, config_resnet50_path, config_efficientnet_path, 
         out_baseline="Results/unet_baseline", 
         out_resnet50="Results/unet_resnet50", 
         out_efficientnet="Results/unet_efficientnet_b1"):
    import yaml
    
    # Load all 3 configs
    with open(config_baseline_path) as f: cfg_baseline = yaml.safe_load(f)
    with open(config_resnet50_path) as f: cfg_resnet50 = yaml.safe_load(f)
    with open(config_efficientnet_path) as f: cfg_efficientnet = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load all 3 models (some may be None if not trained yet)
    print("Loading models...")
    model_baseline = load_model(cfg_baseline, device)
    epoch_baseline = get_best_epoch(cfg_baseline, device) if model_baseline else None
    if model_baseline:
        print(f"✓ Baseline U-Net loaded (Epoch: {epoch_baseline})")
    
    model_resnet50 = load_model(cfg_resnet50, device)
    epoch_resnet50 = get_best_epoch(cfg_resnet50, device) if model_resnet50 else None
    if model_resnet50:
        print(f"✓ U-Net + ResNet50 loaded (Epoch: {epoch_resnet50})")
    
    model_efficientnet = load_model(cfg_efficientnet, device)
    epoch_efficientnet = get_best_epoch(cfg_efficientnet, device) if model_efficientnet else None
    if model_efficientnet:
        print(f"✓ U-Net + EfficientNet-B1 loaded (Epoch: {epoch_efficientnet})")
    print()

    # Load test dataset
    ds = FloodDataset(cfg_baseline["split_json"], split="test", use_preprocessed=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # CSV collectors for all 3 models
    rows_baseline = []
    rows_resnet50 = []
    rows_efficientnet = []
    
    # Create output directories for available models
    if model_baseline:
        os.makedirs(out_baseline, exist_ok=True)
    if model_resnet50:
        os.makedirs(out_resnet50, exist_ok=True)
    if model_efficientnet:
        os.makedirs(out_efficientnet, exist_ok=True)

    print("Generating visualizations for available models...")
    for img, mask, ids in loader:
        img = img.to(device)
        base = ids[0]

        orig_path = next(e["image"] for e in ds.entries if e["id"] == base)
        mask_path = next(e["mask"] for e in ds.entries if e["id"] == base)

        orig = Image.open(orig_path).convert("RGB")
        gt = Image.open(mask_path).convert("L")
        
        # Resize ground truth ke 256x256 untuk match dengan prediction
        gt_resized = gt.resize((256, 256), Image.NEAREST)
        gt_arr = (np.array(gt_resized) > 127).astype(np.uint8)

        # ========== Baseline U-Net Prediction ==========
        if model_baseline:
            with torch.no_grad():
                p_baseline = torch.sigmoid(model_baseline(img)).cpu().numpy()[0,0]
            p_baseline_bin = arr_mask(p_baseline)
            p_baseline_img = Image.fromarray((p_baseline_bin*255).astype("uint8"))
            err_baseline = make_error_map(gt_arr, p_baseline_bin)
            
            out_path_baseline = os.path.join(out_baseline, base + ".png")
            grid_4(orig, gt_resized, p_baseline_img, err_baseline, out_path_baseline, 
                   ["Original", "Mask", "Baseline Unet", "Error"])
            
            iou_baseline = compute_iou(gt_arr, p_baseline_bin)
            dice_baseline = compute_dice(gt_arr, p_baseline_bin)
            rows_baseline.append({"id": base, "epoch": epoch_baseline, "iou": iou_baseline, "dice": dice_baseline})

        # ========== ResNet50 Prediction ==========
        if model_resnet50:
            with torch.no_grad():
                p_resnet50 = torch.sigmoid(model_resnet50(img)).cpu().numpy()[0,0]
            p_resnet50_bin = arr_mask(p_resnet50)
            p_resnet50_img = Image.fromarray((p_resnet50_bin*255).astype("uint8"))
            err_resnet50 = make_error_map(gt_arr, p_resnet50_bin)
            
            out_path_resnet50 = os.path.join(out_resnet50, base + ".png")
            grid_4(orig, gt_resized, p_resnet50_img, err_resnet50, out_path_resnet50,
                   ["Original", "Mask", "ResNet50", "Error"])
            
            iou_resnet50 = compute_iou(gt_arr, p_resnet50_bin)
            dice_resnet50 = compute_dice(gt_arr, p_resnet50_bin)
            rows_resnet50.append({"id": base, "epoch": epoch_resnet50, "iou": iou_resnet50, "dice": dice_resnet50})

        # ========== EfficientNet-B1 Prediction ==========
        if model_efficientnet:
            with torch.no_grad():
                p_efficientnet = torch.sigmoid(model_efficientnet(img)).cpu().numpy()[0,0]
            p_efficientnet_bin = arr_mask(p_efficientnet)
            p_efficientnet_img = Image.fromarray((p_efficientnet_bin*255).astype("uint8"))
            err_efficientnet = make_error_map(gt_arr, p_efficientnet_bin)
            
            out_path_efficientnet = os.path.join(out_efficientnet, base + ".png")
            grid_4(orig, gt_resized, p_efficientnet_img, err_efficientnet, out_path_efficientnet,
                   ["Original", "Mask", "EfficientNetB1", "Error"])
            
            iou_efficientnet = compute_iou(gt_arr, p_efficientnet_bin)
            dice_efficientnet = compute_dice(gt_arr, p_efficientnet_bin)
            rows_efficientnet.append({"id": base, "epoch": epoch_efficientnet, "iou": iou_efficientnet, "dice": dice_efficientnet})

    # Save CSV for available models
    if model_baseline:
        csv_path_baseline = os.path.join(out_baseline, "baseline_metrics.csv")
        with open(csv_path_baseline, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "epoch", "iou", "dice"])
            w.writeheader()
            w.writerows(rows_baseline)

    if model_resnet50:
        csv_path_resnet50 = os.path.join(out_resnet50, "resnet50_metrics.csv")
        with open(csv_path_resnet50, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "epoch", "iou", "dice"])
            w.writeheader()
            w.writerows(rows_resnet50)

    if model_efficientnet:
        csv_path_efficientnet = os.path.join(out_efficientnet, "efficientnet_metrics.csv")
        with open(csv_path_efficientnet, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "epoch", "iou", "dice"])
            w.writeheader()
            w.writerows(rows_efficientnet)

    # Print results for available models
    print()
    if model_baseline:
        print(f"✓ Baseline U-Net visualizations saved to: {out_baseline}")
    if model_resnet50:
        print(f"✓ ResNet50 visualizations saved to: {out_resnet50}")
    if model_efficientnet:
        print(f"✓ EfficientNet-B1 visualizations saved to: {out_efficientnet}")
    
    avg_iou_baseline = sum(r["iou"] for r in rows_baseline) / len(rows_baseline) if rows_baseline else 0
    avg_iou_resnet50 = sum(r["iou"] for r in rows_resnet50) / len(rows_resnet50) if rows_resnet50 else 0
    avg_iou_efficientnet = sum(r["iou"] for r in rows_efficientnet) / len(rows_efficientnet) if rows_efficientnet else 0
    
    print(f"\n{'='*60}")
    print(f"AVERAGE IoU COMPARISON")
    print(f"{'='*60}")
    if model_baseline:
        print(f"Baseline U-Net:          {avg_iou_baseline:.4f}")
    if model_resnet50:
        print(f"U-Net + ResNet50:        {avg_iou_resnet50:.4f}")
    if model_efficientnet:
        print(f"U-Net + EfficientNet-B1: {avg_iou_efficientnet:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_baseline", required=True, help="Path to Baseline U-Net config YAML")
    parser.add_argument("--config_resnet50", required=True, help="Path to ResNet50 config YAML")
    parser.add_argument("--config_efficientnet", required=True, help="Path to EfficientNet config YAML")
    parser.add_argument("--out_baseline", default="Results/unet_baseline", help="Output directory for Baseline results")
    parser.add_argument("--out_resnet50", default="Results/unet_resnet50", help="Output directory for ResNet50 results")
    parser.add_argument("--out_efficientnet", default="Results/unet_efficientnet_b1", help="Output directory for EfficientNet results")
    args = parser.parse_args()

    main(args.config_baseline, args.config_resnet50, args.config_efficientnet,
         args.out_baseline, args.out_resnet50, args.out_efficientnet)
