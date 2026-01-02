import torch
from tqdm import tqdm

def iou_score(pred, target, thr=0.5, eps=1e-7):
    pred_bin = (torch.sigmoid(pred) > thr).float()
    inter = (pred_bin * target).sum(dim=(1,2,3))
    union = (pred_bin + target - pred_bin*target).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def dice_score(pred, target, thr=0.5, eps=1e-7):
    pred_bin = (torch.sigmoid(pred) > thr).float()
    inter = (pred_bin * target).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

def compute_metrics(model, loader, device, thr=0.5, eps=1e-7):
    """
    Compute comprehensive metrics: IoU, Dice, Accuracy, Precision, Recall, F1
    """
    model.eval()
    
    total_iou = 0.0
    total_dice = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Computing metrics', colour='cyan', ncols=100)
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            preds = model(imgs)
            pred_bin = (torch.sigmoid(preds) > thr).float()
            
            # Per batch metrics
            TP = (pred_bin * masks).sum(dim=(1,2,3))
            FP = (pred_bin * (1 - masks)).sum(dim=(1,2,3))
            FN = ((1 - pred_bin) * masks).sum(dim=(1,2,3))
            TN = ((1 - pred_bin) * (1 - masks)).sum(dim=(1,2,3))
            
            # IoU
            inter = TP
            union = TP + FP + FN
            iou = (inter + eps) / (union + eps)
            
            # Dice
            dice = (2 * inter + eps) / (2 * inter + FP + FN + eps)
            
            # Accuracy
            accuracy = (TP + TN + eps) / (TP + TN + FP + FN + eps)
            
            # Precision
            precision = (TP + eps) / (TP + FP + eps)
            
            # Recall
            recall = (TP + eps) / (TP + FN + eps)
            
            # F1
            f1 = 2 * (precision * recall) / (precision + recall + eps)
            
            total_iou += iou.mean().item()
            total_dice += dice.mean().item()
            total_accuracy += accuracy.mean().item()
            total_precision += precision.mean().item()
            total_recall += recall.mean().item()
            total_f1 += f1.mean().item()
            count += 1
            
            # Update progress bar
            pbar.set_postfix({'iou': f'{total_iou/count:.4f}', 'f1': f'{total_f1/count:.4f}'})
    
    return {
        'iou': total_iou / count if count > 0 else 0.0,
        'dice': total_dice / count if count > 0 else 0.0,
        'accuracy': total_accuracy / count if count > 0 else 0.0,
        'precision': total_precision / count if count > 0 else 0.0,
        'recall': total_recall / count if count > 0 else 0.0,
        'f1': total_f1 / count if count > 0 else 0.0
    }
