import torch, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from tqdm import tqdm
from src.training.losses import bce_dice_loss, focal_dice_loss
from src.training.metrics import iou_score

def train_one_epoch(model, loader, optimizer, device, scaler=None, accumulation_steps=1, loss_fn='bce_dice', loss_cfg=None):
    """
    Train one epoch with configurable loss function
    loss_fn: 'bce_dice' (default) or 'focal_dice'
    loss_cfg: dict with loss hyperparameters
    """
    model.train()
    running_loss = 0.0
    
    # Setup loss function
    if loss_cfg is None:
        loss_cfg = {}
    
    pbar = tqdm(loader, desc='Training', colour='green', ncols=100)
    for i, (imgs, masks, _) in enumerate(pbar):
        imgs = imgs.to(device); masks = masks.to(device)
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            preds = model(imgs)
            
            # Choose loss function
            if loss_fn == 'focal_dice':
                loss = focal_dice_loss(
                    preds, masks,
                    alpha=loss_cfg.get('focal_alpha', 0.25),
                    gamma=loss_cfg.get('focal_gamma', 2.0),
                    weight=loss_cfg.get('bce_dice_weight', 0.5)
                )
            else:
                loss = bce_dice_loss(
                    preds, masks,
                    bce_weight=loss_cfg.get('bce_dice_weight', 0.5)
                )
            
            loss = loss / accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        else:
            loss.backward()
            if (i+1) % accumulation_steps == 0:
                optimizer.step(); optimizer.zero_grad()
        running_loss += loss.item() * accumulation_steps
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{running_loss/(i+1):.4f}'})
    
    return running_loss / (len(loader) if len(loader)>0 else 1)

def validate(model, loader, device):
    model.eval()
    ious = []
    pbar = tqdm(loader, desc='Validation', colour='blue', ncols=100)
    with torch.no_grad():
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device); masks = masks.to(device)
            preds = model(imgs)
            iou = iou_score(preds, masks)
            ious.append(iou)
            
            # Update progress bar with current IoU
            pbar.set_postfix({'iou': f'{sum(ious)/len(ious):.4f}'})
    
    return sum(ious)/len(ious) if len(ious)>0 else 0.0
