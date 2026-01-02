#!/usr/bin/env python3
"""
Create stratified splits based on flood pixel ratio per image.

Usage:
    python src/preprocessing/create_splits.py --images data/raw/Images --masks data/raw/Masks --out data/splits/splits.json
"""

import os, json, argparse
from glob import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def compute_ratio(mask_path):
    m = np.array(Image.open(mask_path).convert('L'))
    return (m > 127).sum() / m.size


def make_bins(ratios, edges=[0.0, 0.001, 0.05, 0.2, 0.5, 1.0]):
    strata = np.digitize(ratios, edges, right=True) - 1
    return strata.tolist(), edges


def balance_strata(strata):
    """Merge bins with too few samples to avoid stratification errors"""
    from collections import Counter
    counts = Counter(strata)
    
    # If any bin has less than 2 samples, merge with adjacent bins
    merged = []
    for s in strata:
        if counts[s] < 2:
            # Merge to most common bin
            s = counts.most_common(1)[0][0]
        merged.append(s)
    return merged


def main(images_dir, masks_dir, out_path, seed=42):
    image_paths = sorted(glob(os.path.join(images_dir, '*')))
    mask_paths = sorted(glob(os.path.join(masks_dir, '*')))

    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}

    keys = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
    print(f"Found {len(keys)} pairs.")

    ratios = [compute_ratio(mask_map[k]) for k in keys]
    strata, edges = make_bins(ratios)
    
    # Balance strata to avoid too few samples in any bin
    strata = balance_strata(strata)
    
    from collections import Counter
    print(f"Strata distribution: {dict(Counter(strata))}")

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
        train_idx, temp_idx = next(sss1.split(keys, strata))

        train_keys = [keys[i] for i in train_idx]
        temp_keys = [keys[i] for i in temp_idx]
        temp_strata = [strata[i] for i in temp_idx]

        val_prop = val_ratio / (val_ratio + test_ratio)

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_prop), random_state=seed + 1)
        val_idx_rel, test_idx_rel = next(sss2.split(temp_keys, temp_strata))

        val_keys = [temp_keys[i] for i in val_idx_rel]
        test_keys = [temp_keys[i] for i in test_idx_rel]
    except ValueError as e:
        # Fallback to random split if stratification fails
        print(f"Warning: Stratification failed ({e}). Using random split instead.")
        np.random.seed(seed)
        indices = np.random.permutation(len(keys))
        
        n_train = int(len(keys) * train_ratio)
        n_val = int(len(keys) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        train_keys = [keys[i] for i in train_idx]
        val_keys = [keys[i] for i in val_idx]
        test_keys = [keys[i] for i in test_idx]

    def to_list(ks):
        return [{'id': k, 'image': img_map[k], 'mask': mask_map[k]} for k in ks]

    out = {
        'train': to_list(train_keys),
        'val': to_list(val_keys),
        'test': to_list(test_keys),
        'meta': {'bin_edges': edges}
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n✓ Saved splits to: {out_path}")
    print(f"  Train: {len(train_keys)} images ({len(train_keys)/len(keys)*100:.1f}%)")
    print(f"  Val:   {len(val_keys)} images ({len(val_keys)/len(keys)*100:.1f}%)")
    print(f"  Test:  {len(test_keys)} images ({len(test_keys)/len(keys)*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='data/raw/Image')
    parser.add_argument('--masks', default='data/raw/Mask')
    parser.add_argument('--out', default='data/splits/splits.json')
    args = parser.parse_args()
    main(args.images, args.masks, args.out)
