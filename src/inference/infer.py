#!/usr/bin/env python3
"""
Inference script for U-Net models with different encoders.
Supports: baseline, resnet50, efficientnet_b1
"""
import os, sys, yaml, argparse, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.models.unet import UNet
from src.Dataset.loader import FloodDataset
from torch.utils.data import DataLoader

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_pred_grid(orig_path, gt_path, pred_arr, out_path, iou=None):
    img = Image.open(orig_path).convert('RGB').resize((256,256))
    gt = Image.open(gt_path).convert('L').resize((256,256))
    pred_img = Image.fromarray((pred_arr*255).astype('uint8')).convert('L').resize((256,256))

    w,h = img.size
    new = Image.new('RGB', (w*3, h))
    new.paste(img, (0,0))
    new.paste(Image.merge('RGB', (gt,gt,gt)), (w,0))
    new.paste(Image.merge('RGB', (pred_img,pred_img,pred_img)), (w*2,0))

    draw = ImageDraw.Draw(new)
    try:
        fnt = ImageFont.truetype('DejaVuSans.ttf', 14)
    except:
        fnt = None
    if iou is not None:
        draw.text((w*2+8,8), f'IoU: {iou:.3f}', fill=(255,255,255), font=fnt)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new.save(out_path)

def run_infer(config_path):
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get model configuration
    base_c = cfg.get('base_c', 32)
    encoder = cfg.get('encoder', 'baseline')
    
    # Build model with specified encoder
    model = UNet(in_ch=3, n_classes=1, base_c=base_c, encoder=encoder)
    ckpt = os.path.join(cfg.get('checkpoint_dir','Save_models'), cfg.get('save_name','best_unet.pth'))

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model weights not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.to(device); model.eval()

    ds = FloodDataset(cfg['split_json'], split='test', transforms=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    metrics = []
    print(f"Running inference with {encoder} encoder...")
    for img, mask, idd in loader:
        img = img.to(device); mask = mask.to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img)).cpu().numpy()[0,0]
        base = idd[0]
        entry = next(e for e in ds.entries if e['id']==base)
        orig = entry['image']; gt = entry['mask']

        pred_bin = (pred > 0.5).astype('uint8')
        gt_arr = (np.array(Image.open(gt).convert('L')) > 127).astype('uint8')
        inter = (pred_bin & gt_arr).sum()
        union = (pred_bin | gt_arr).sum()
        iou = float(inter/union) if union>0 else 0.0
        
        # Calculate Dice
        dice = float(2*inter / (2*inter + (pred_bin ^ gt_arr).sum())) if (2*inter + (pred_bin ^ gt_arr).sum()) > 0 else 0.0
        
        # Get output directory from metrics_out
        metrics_out = cfg.get('metrics_out', f'Results/{encoder}/metrics.csv')
        out_dir = os.path.dirname(metrics_out)
        pred_dir = os.path.join(out_dir, 'predictions')
        outp = os.path.join(pred_dir, base + '.png')
        
        save_pred_grid(orig, gt, pred, outp, iou=iou)
        metrics.append({'id': base, 'iou': iou, 'dice': dice})
    
    # Save metrics as JSON
    out_metrics = os.path.join(out_dir, 'predictions_metrics.json')
    os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
    with open(out_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    avg_iou = sum(m['iou'] for m in metrics) / len(metrics) if metrics else 0
    print(f'Saved predictions and metrics to: {out_dir}')
    print(f'Average IoU: {avg_iou:.4f}')
