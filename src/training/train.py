#!/usr/bin/env python3
import os, sys, yaml, argparse, torch, json, csv
import random
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from torch.utils.data import DataLoader
from src.Dataset.loader import FloodDataset
from src.Dataset.transform import default_transforms
from src.models.unet import UNet
from src.training.engine import train_one_epoch, validate
from src.training.metrics import compute_metrics
from torch.optim import AdamW, SGD, Adam
from torch.cuda import amp

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed} for reproducibility")

def worker_init_fn(worker_id):
    """Initialize worker with unique seed for reproducibility"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

def build_model(cfg):
    """
    Build model based on encoder type:
    - baseline: Standard U-Net
    - resnet50: U-Net with ResNet50 encoder
    - efficientnet_b1: U-Net with EfficientNet-B1 encoder
    """
    encoder = cfg.get('encoder', 'baseline')
    base_c = cfg.get('base_c', 32)
    return UNet(in_ch=3, n_classes=1, base_c=base_c, encoder=encoder)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load checkpoint untuk resume training.
    Returns: (start_epoch, best_val_iou)
    """
    if not os.path.exists(checkpoint_path):
        return 1, 0.0
    
    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    
    print(f"✓ Checkpoint loaded: Resume from epoch {start_epoch}, Best IoU: {best_val_iou:.4f}")
    return start_epoch, best_val_iou

def save_checkpoint(checkpoint_path, epoch, model, optimizer, best_val_iou, scheduler=None):
    """
    Save checkpoint untuk training progress.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_iou': best_val_iou,
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: epoch {epoch}, IoU {best_val_iou:.4f}")

def main(cfg):
    
    # NO SEED to match OFAT (comment out for exact replication)
    # seed = cfg.get('seed', 42)
    # set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, using CPU (training will be slow!)")
    print(f"{'='*60}\n")

    # Load datasets - NO TRANSFORMS to match OFAT setup
    train_ds = FloodDataset(cfg['split_json'], split='train', transforms=None, use_preprocessed=True)
    val_ds = FloodDataset(cfg['split_json'], split='val', transforms=None, use_preprocessed=False)
    test_ds = FloodDataset(cfg['split_json'], split='test', transforms=None, use_preprocessed=False)
    
    # DataLoader with worker seed for reproducibility
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size',4), shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size',4), shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.get('batch_size',4), shuffle=False, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    model = build_model(cfg).to(device)
    opt_name = cfg.get('optimizer','adamw').lower()
    lr = cfg.get('lr', 1e-3)
    
    if opt_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif opt_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # Learning rate scheduler
    scheduler = None
    if cfg.get('use_scheduler', False):
        scheduler_type = cfg.get('scheduler_type', 'cosine')
        warmup_epochs = cfg.get('warmup_epochs', 5)
        epochs = cfg.get('epochs', 20)
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=lr*0.01)
    
    scaler = amp.GradScaler() if cfg.get('use_amp', True) and torch.cuda.is_available() else None

    best_val_iou = 0.0
    best_val_dice = 0.0
    best_epoch = 0
    best_val_score = 0.0  # gabungan iou+dice
    epochs = cfg.get('epochs', 20)
    batch_size = cfg.get('batch_size', 4)
    checkpoint_dir = cfg.get('checkpoint_dir','Save_models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint path untuk resume
    checkpoint_path = os.path.join(checkpoint_dir, 'training_checkpoint.pth')
    start_epoch = 1
    
    # Cek jika ada checkpoint untuk resume
    if cfg.get('resume', False) and os.path.exists(checkpoint_path):
        start_epoch, best_val_iou = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"🔄 Resuming training from epoch {start_epoch}")
    else:
        print(f"🆕 Starting fresh training")
    
    # Display training configuration
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {cfg.get('model', 'unet')}")
    print(f"Encoder: {cfg.get('encoder', 'baseline')}")
    print(f"Base channels: {cfg.get('base_c', 32)}")
    print(f"Batch size: {batch_size} (accum_steps: {cfg.get('accum_steps', 1)})")
    print(f"Learning rate: {lr}")
    print(f"Optimizer: {opt_name}")
    print(f"Total epochs: {epochs}")
    print(f"Loss function: {cfg.get('loss_type', 'bce_dice')}")
    if cfg.get('loss_type') == 'focal_dice':
        print(f"  - Focal alpha: {cfg.get('focal_alpha', 0.25)}")
        print(f"  - Focal gamma: {cfg.get('focal_gamma', 2.0)}")
    print(f"Scheduler: {cfg.get('scheduler_type', 'none') if cfg.get('use_scheduler', False) else 'none'}")
    if cfg.get('use_scheduler', False):
        print(f"  - Warmup epochs: {cfg.get('warmup_epochs', 5)}")
    print(f"AMP enabled: {scaler is not None}")
    print(f"{'='*60}\n")
    
    # CSV logging per epoch dengan full metrics
    epoch_log_path = os.path.join(cfg.get('checkpoint_dir','Save_models'), 'training_log.csv')
    with open(epoch_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_iou', 'val_dice', 'val_accuracy', 
            'val_precision', 'val_recall', 'val_f1', 'learning_rate'
        ])
    
    warmup_epochs = cfg.get('warmup_epochs', 5) if scheduler else 0
    
    # Loss configuration
    loss_fn = cfg.get('loss_type', 'bce_dice')
    loss_cfg = {
        'focal_alpha': cfg.get('focal_alpha', 0.25),
        'focal_gamma': cfg.get('focal_gamma', 2.0),
        'bce_dice_weight': cfg.get('bce_dice_weight', 0.5)
    }
    
    import time
    epoch_times = []
    total_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, 
                                    accumulation_steps=cfg.get('accum_steps',1),
                                    loss_fn=loss_fn, loss_cfg=loss_cfg)
        # Compute validation metrics (IoU, Dice, Accuracy, Precision, Recall, F1)
        val_metrics = compute_metrics(model, val_loader, device)
        val_iou = val_metrics['iou']
        # Learning rate scheduling with warmup
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if epoch > warmup_epochs:
                scheduler.step()
            else:
                # Linear warmup
                warmup_lr = lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                current_lr = warmup_lr
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_iou={val_iou:.4f} val_dice={val_metrics['dice']:.4f} lr={current_lr:.6f} time={epoch_time:.2f}s")
        # Log per epoch dengan full metrics
        with open(epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                round(train_loss, 4), 
                round(val_iou, 4),
                round(val_metrics.get('dice', 0), 4),
                round(val_metrics.get('accuracy', 0), 4),
                round(val_metrics.get('precision', 0), 4),
                round(val_metrics.get('recall', 0), 4),
                round(val_metrics.get('f1', 0), 4),
                round(current_lr, 6)
            ])
        val_dice = val_metrics.get('dice', 0)
        val_score = (val_iou + val_dice) / 2
        if val_score > best_val_score:
            best_val_iou = val_iou
            best_val_dice = val_dice
            best_val_score = val_score
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, cfg.get('save_name','best_model.pth')))
            # Save checkpoint untuk resume
            save_checkpoint(checkpoint_path, epoch, model, optimizer, best_val_iou, scheduler)
        # Auto-save checkpoint setiap 5 epoch
        if epoch % 5 == 0:
            save_checkpoint(checkpoint_path, epoch, model, optimizer, best_val_iou, scheduler)
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.2f} s")
    print(f"Average epoch time: {np.mean(epoch_times):.2f} s")
    # Simpan waktu ke file
    with open(os.path.join(checkpoint_dir, 'training_time_log.txt'), 'a') as f:
        f.write(f"Total training time: {total_time:.2f} s\n")
        f.write(f"Average epoch time: {np.mean(epoch_times):.2f} s\n")
        f.write(f"Epoch times: {epoch_times}\n")
    # Compute test metrics
    print("\nEvaluating on test set...")
    test_metrics = compute_metrics(model, test_loader, device)
    
    # Save comprehensive summary CSV with all configuration
    summary_path = cfg.get('metrics_out', 'Results/metrics_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'encoder', 'base_c', 'optimizer', 'batch_size', 'learning_rate', 'accum_steps',
            'loss_type', 'focal_alpha', 'focal_gamma', 'bce_dice_weight',
            'use_scheduler', 'scheduler_type', 'warmup_epochs',
            'use_amp', 'total_epochs', 'best_epoch', 'best_val_iou', 'best_val_dice',
            'test_iou', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_dice'
        ])
        writer.writerow([
            cfg.get('model', 'unet'),
            cfg.get('encoder', 'baseline'),
            cfg.get('base_c', 32),
            opt_name,
            cfg.get('batch_size', 4),
            lr,
            cfg.get('accum_steps', 1),
            cfg.get('loss_type', 'bce_dice'),
            cfg.get('focal_alpha', 0.25),
            cfg.get('focal_gamma', 2.0),
            cfg.get('bce_dice_weight', 0.5),
            cfg.get('use_scheduler', False),
            cfg.get('scheduler_type', 'none'),
            cfg.get('warmup_epochs', 0),
            cfg.get('use_amp', True),
            epochs,
            best_epoch,
            round(best_val_iou, 4),
            round(best_val_dice, 4),
            round(test_metrics.get('iou', 0), 4),
            round(test_metrics.get('accuracy', 0), 4),
            round(test_metrics.get('precision', 0), 4),
            round(test_metrics.get('recall', 0), 4),
            round(test_metrics.get('f1', 0), 4),
            round(test_metrics.get('dice', 0), 4)
        ])
    
    print(f"\nTraining complete!")
    print(f"Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")
    print(f"Test metrics: IoU={test_metrics.get('iou', 0):.4f}, F1={test_metrics.get('f1', 0):.4f}")
    print(f"Results saved to: {summary_path}")
    
    # Hapus checkpoint file karena training selesai
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🗑️  Checkpoint file removed (training complete)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/experiments/Config/unet.yaml')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results (model, metrics, log)')
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    cfg = load_config(args.config)

    # Override output_dir if given
    if args.output_dir:
        # Update all output paths in config
        cfg['checkpoint_dir'] = args.output_dir
        cfg['metrics_out'] = os.path.join(args.output_dir, 'metrics_summary.csv')
        cfg['save_name'] = 'best_model.pth'
    # Save seed info in config for traceability
    cfg['seed'] = args.seed

        # Run main training with config dict
    main(cfg)
    # If output_dir is set, main() should use the updated cfg
