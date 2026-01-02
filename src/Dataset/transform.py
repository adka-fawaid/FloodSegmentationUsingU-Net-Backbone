# Enhanced data augmentation for 70-80% accuracy target
import random
import torch
import torchvision.transforms.functional as TF

def default_transforms(img, mask, training=True):
    """
    img: torch tensor CHW
    mask: torch tensor 1HW
    returns transformed (img, mask)
    Enhanced augmentation for better generalization
    """
    if training:
        # Horizontal flip (50%)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        
        # Vertical flip (50%)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        
        # Random rotation 90, 180, 270 degrees (60%)
        if random.random() > 0.4:
            angles = [90, 180, 270]
            angle = random.choice(angles)
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        
        # Brightness jitter (70%)
        if random.random() > 0.3:
            factor = 0.7 + random.random() * 0.6  # [0.7, 1.3]
            img = img * factor
            img = img.clamp(0, 1)
        
        # Contrast jitter (50%)
        if random.random() > 0.5:
            factor = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            mean = img.mean(dim=[1, 2], keepdim=True)
            img = (img - mean) * factor + mean
            img = img.clamp(0, 1)
        
        # Gaussian noise (30%)
        if random.random() > 0.7:
            noise = torch.randn_like(img) * 0.02
            img = img + noise
            img = img.clamp(0, 1)
    
    return img, mask
