import torch
import torch.nn.functional as F

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss untuk handle class imbalance
    pred: raw logits (B,1,H,W)
    target: 0/1 float (B,1,H,W)
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pred_prob = torch.sigmoid(pred)
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    focal = focal_weight * bce
    return focal.mean()

def dice_loss(pred, target):
    """
    Dice Loss untuk overlap optimization
    pred: raw logits (B,1,H,W)
    target: 0/1 float (B,1,H,W)
    """
    pred_sig = torch.sigmoid(pred)
    smooth = 1.0
    inter = (pred_sig * target).sum(dim=(1,2,3))
    union = pred_sig.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    """
    Standard BCE + Dice Loss (backward compatibility)
    pred: raw logits (B,1,H,W)
    target: 0/1 float (B,1,H,W)
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice

def focal_dice_loss(pred, target, alpha=0.25, gamma=2.0, weight=0.5):
    """
    Focal + Dice Loss untuk akurasi maksimal (70-80%+)
    pred: raw logits (B,1,H,W)
    target: 0/1 float (B,1,H,W)
    """
    focal = focal_loss(pred, target, alpha, gamma)
    dice = dice_loss(pred, target)
    return weight * focal + (1 - weight) * dice
