import os, json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class FloodDataset(Dataset):
    """
    Dataset for flood segmentation.
    - For TRAIN: reads from data/processed/train/images_preprocessed/
    - For VAL/TEST: reads from data/raw/ (original splits.json paths)
    Returns: img (CHW float32 0..1), mask (1HW float32 0/1), id (basename)
    """
    def __init__(self, split_json, split='train', transforms=None, use_preprocessed=True, processed_dir='Data/processed'):
        with open(split_json) as f:
            sp = json.load(f)
        if split not in sp:
            raise ValueError(f"Split {split} not in {split_json}")
        self.entries = sp[split]
        self.transforms = transforms
        self.split = split
        self.use_preprocessed = use_preprocessed
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        
        # For TRAIN split, use preprocessed data if available
        if self.split == 'train' and self.use_preprocessed:
            base = e['id']
            img_p = os.path.join(self.processed_dir, 'train', 'images', base + '.png')
            mask_p = os.path.join(self.processed_dir, 'train', 'masks', base + '.png')
            
            # Fallback to original if preprocessed not found
            if not os.path.exists(img_p):
                print(f"Warning: Preprocessed image not found for {base}, using original")
                img_p = e['image']
                mask_p = e['mask']
        else:
            # For val/test, use original paths from splits.json
            img_p = e['image']
            mask_p = e['mask']
        
        img = Image.open(img_p).convert('RGB')
        mask = Image.open(mask_p).convert('L')
        
        # Ensure consistent size (resize to 256x256)
        img = img.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        img = np.array(img).astype('float32') / 255.0
        mask = (np.array(mask) > 127).astype('float32')

        # HWC -> CHW
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask, e['id']
