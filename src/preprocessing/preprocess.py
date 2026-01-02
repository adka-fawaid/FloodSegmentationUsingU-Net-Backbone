#!/usr/bin/env python3
"""
Apply preprocessing to images:
- resize
- CLAHE
- gamma correction
- unsharp masking

ONLY processes TRAIN split and saves:
- Original images (resized)
- Preprocessed images
"""

import os, json, argparse, shutil
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import cv2


def apply_clahe(img):
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    lab = cv2.merge((cl,a,b))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def gamma_correction(img, gamma=1.0):
    inv = 1.0 / gamma
    lut = [min(255, int((i / 255.0)**inv * 255.0)) for i in range(256)]
    return img.point(lut * 3)


def unsharp(img, radius=1, percent=150, threshold=3):
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def process(entry, out_comparison_dir, out_preprocessed_dir, out_mask_dir, size=256):
    img = Image.open(entry['image']).convert('RGB')
    mask = Image.open(entry['mask']).convert('L')

    # Resize original
    img_resized = img.resize((size,size), Image.BILINEAR)

    # Apply preprocessing
    img_preprocessed = apply_clahe(img_resized)
    img_preprocessed = gamma_correction(img_preprocessed, 1.0)
    img_preprocessed = unsharp(img_preprocessed)

    # Create directories
    os.makedirs(out_comparison_dir, exist_ok=True)
    os.makedirs(out_preprocessed_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(entry['image']))[0]
    
    # Create side-by-side comparison image with labels on top
    from PIL import ImageDraw, ImageFont
    w, h = img_resized.size
    header_height = 30
    
    # Create canvas with extra space for text header
    comparison = Image.new('RGB', (w*2, h + header_height), color=(255, 255, 255))
    
    # Paste images below header
    comparison.paste(img_resized, (0, header_height))
    comparison.paste(img_preprocessed, (w, header_height))
    
    # Add labels centered above each image
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = None
    
    # Calculate text positions (centered)
    text1 = "Original"
    text2 = "Preprocessed"
    
    if font:
        bbox1 = draw.textbbox((0, 0), text1, font=font)
        bbox2 = draw.textbbox((0, 0), text2, font=font)
        text1_width = bbox1[2] - bbox1[0]
        text2_width = bbox2[2] - bbox2[0]
    else:
        text1_width = len(text1) * 6
        text2_width = len(text2) * 8
    
    draw.text((w//2 - text1_width//2, 5), text1, fill=(0, 0, 0), font=font)
    draw.text((w + w//2 - text2_width//2, 5), text2, fill=(0, 0, 0), font=font)
    
    # Save comparison image
    comparison.save(os.path.join(out_comparison_dir, base + '.png'))
    
    # Save preprocessed only (for training)
    img_preprocessed.save(os.path.join(out_preprocessed_dir, base + '.png'))
    
    # Copy mask as-is (no processing, just copy)
    shutil.copy(entry['mask'], os.path.join(out_mask_dir, base + '.png'))


def main(split_path, out_dir, size=256):

    with open(split_path) as f:
        splits = json.load(f)

    # ONLY process TRAIN split
    print("Processing TRAIN split only...")
    pbar = tqdm(splits['train'], desc='Preprocessing', colour='green', ncols=100)
    for entry in pbar:
        comparison_dir = os.path.join(out_dir, 'train', 'comparison')
        preprocessed_dir = os.path.join(out_dir, 'train', 'images')
        mask_dir = os.path.join(out_dir, 'train', 'masks')
        process(entry, comparison_dir, preprocessed_dir, mask_dir, size=size)
        pbar.set_postfix({'file': entry['id']})

    print("\n✓ Preprocessing complete (train only).")
    print(f"Saved to: {out_dir}/train/")
    print(f"  - comparison/  (Original | Preprocessed side-by-side)")
    print(f"  - images/      (Preprocessed images for training)")
    print(f"  - masks/       (Copied masks, no processing)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='data/splits/splits.json')
    parser.add_argument('--out', default='data/processed')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    main(args.split, args.out, args.size)
