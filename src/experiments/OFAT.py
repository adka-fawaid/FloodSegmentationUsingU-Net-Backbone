import random
#!/usr/bin/env python3
"""
OFAT Hyperparameter Optimization untuk U-Net models.
Script mandiri yang mengoptimasi hyperparameter satu per satu.
Supports: Baseline U-Net, ResNet50, EfficientNet-B1 encoders.
"""
import os, yaml, json, argparse, csv
import torch
from torch.utils.data import DataLoader
import sys

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from src.Dataset.loader import FloodDataset
from src.models.unet import UNet
from src.training.engine import train_one_epoch
from src.training.metrics import compute_metrics
from torch.optim import AdamW, SGD, Adam
from torch.cuda import amp

def check_phase_completion(output_dir, phase_name):
    """
    Cek apakah fase tertentu sudah selesai berdasarkan CSV file.
    Returns: (is_complete, best_value, best_iou)
    """
    csv_path = os.path.join(output_dir, f"{phase_name}.csv")
    if not os.path.exists(csv_path):
        return False, None, 0.0
    
    try:
        # Baca CSV dan cari nilai terbaik berdasarkan final epoch (konsisten dengan optimize_parameter)
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return False, None, 0.0
        
        # Group by param_value, ambil epoch terakhir (final), hitung score
        final_results = []
        for param_val in df['param_value'].unique():
            subset = df[df['param_value'] == param_val]
            # Ambil row dengan epoch tertinggi (final epoch)
            final_row = subset.loc[subset['epoch'].idxmax()]
            final_iou = final_row['val_iou']
            final_dice = final_row['val_dice']
            final_score = (final_iou + final_dice) / 2
            final_results.append({
                'param_value': param_val,
                'final_iou': final_iou,
                'final_dice': final_dice,
                'final_score': final_score
            })
        
        # Pilih param_value dengan score tertinggi
        best = max(final_results, key=lambda x: x['final_score'])
        best_value = best['param_value']
        best_iou = best['final_iou']
        
        return True, best_value, best_iou
    except Exception as e:
        print(f"⚠️  Error reading {csv_path}: {e}")
        return False, None, 0.0


def OFAT_optimize(base_config_path, output_dir='Results/OFAT'):
    # Set seed agar hasil fair antar kombinasi
    import torch
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """
    Melakukan OFAT untuk mencari hyperparameter terbaik.
    Setiap fase menyimpan CSV detail dengan metrik per-epoch.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base config
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tampilkan info device seperti di train.py
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("PERINGATAN: CUDA tidak tersedia, menggunakan CPU (training akan lambat!)")
    print(f"{'='*60}\n")
    
    # Ambil parameter search space dari config YAML
    if 'OFAT' not in base_cfg:
        print("ERROR: 'OFAT' section tidak ditemukan di config YAML!")
        print("Pastikan config memiliki section 'OFAT' dengan optimizer, batch_size, lr, dan epochs")
        sys.exit(1)
    
    param_space = base_cfg['OFAT']
    print(f"Search space dimuat dari config:")
    print(f"  - Optimizer: {param_space['optimizer']}")
    print(f"  - Batch size: {param_space['batch_size']}")
    print(f"  - Learning rate: {len(param_space['lr'])} values")
    print(f"  - Epochs: {param_space['epochs']}\n")
    
    # Initialize with baseline parameters (fixed starting point untuk fair comparison)
    best_params = {
        'optimizer': 'adamw',
        'batch_size': 4,
        'lr': 0.001,
        'epochs': 10  # Default for search phase
    }
    
    print(f"Baseline parameters (starting point for OFAT):")
    print(f"  - Optimizer: {best_params['optimizer']}")
    print(f"  - Batch size: {best_params['batch_size']}")
    print(f"  - Learning rate: {best_params['lr']}")
    print(f"  - Search epochs: {best_params['epochs']}\n")
    
    # Optimize in order: optimizer -> batch_size -> lr -> epochs
    param_order = ['optimizer', 'batch_size', 'lr', 'epochs']
    
    model_type = base_cfg.get('model', 'unet_baseline')
    encoder_type = base_cfg.get('encoder', 'baseline')
    
    # Map encoder to readable name
    encoder_names = {
        'baseline': 'Baseline U-Net',
        'resnet50': 'U-Net + ResNet50',
        'efficientnet_b1': 'U-Net + EfficientNet-B1'
    }
    model_name = encoder_names.get(encoder_type, encoder_type)
    
    print("="*60)
    print(f"ONE-FACTOR-AT-A-TIME OPTIMIZATION - {model_name.upper()}")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Encoder: {encoder_type}")
    print(f"Base channels: {base_cfg.get('base_c', 32)}")
    print(f"Parameter awal: {best_params}")
    print(f"Urutan pencarian: {' -> '.join(param_order)}")
    print("="*60)
    
    # Phase 1: Optimize Optimizer
    print(f"\n{'='*60}")
    print(f"PHASE 1: OPTIMIZING OPTIMIZER")
    print(f"{'='*60}")
    phase1_complete, saved_value, saved_iou = check_phase_completion(output_dir, 'optimizer')
    if phase1_complete:
        print(f"⏭️  PHASE 1 ALREADY COMPLETE - Skipping")
        print(f"   Loaded best optimizer: {saved_value} (IoU: {saved_iou:.4f})")
        best_params['optimizer'] = saved_value
        phase1_data = []  # Empty data since we skipped
    else:
        best_params['optimizer'], phase1_data = optimize_parameter(
            'optimizer', param_space['optimizer'], best_params, base_cfg, device, output_dir, model_type
        )
    
    # Phase 2: Optimize Batch Size
    print(f"\n{'='*60}")
    print(f"PHASE 2: OPTIMIZING BATCH SIZE")
    print(f"{'='*60}")
    phase2_complete, saved_value, saved_iou = check_phase_completion(output_dir, 'batch_size')
    if phase2_complete:
        print(f"⏭️  PHASE 2 ALREADY COMPLETE - Skipping")
        print(f"   Loaded best batch_size: {saved_value} (IoU: {saved_iou:.4f})")
        best_params['batch_size'] = saved_value
        phase2_data = []
    else:
        best_params['batch_size'], phase2_data = optimize_parameter(
            'batch_size', param_space['batch_size'], best_params, base_cfg, device, output_dir, model_type
        )
    
    # Phase 3: Optimize Learning Rate
    print(f"\n{'='*60}")
    print(f"PHASE 3: OPTIMIZING LEARNING RATE")
    print(f"{'='*60}")
    phase3_complete, saved_value, saved_iou = check_phase_completion(output_dir, 'lr')
    if phase3_complete:
        print(f"⏭️  PHASE 3 ALREADY COMPLETE - Skipping")
        print(f"   Loaded best lr: {saved_value} (IoU: {saved_iou:.4f})")
        best_params['lr'] = saved_value
        phase3_data = []
    else:
        best_params['lr'], phase3_data = optimize_parameter(
            'lr', param_space['lr'], best_params, base_cfg, device, output_dir, model_type
        )
    
    # Phase 4: Optimize Epochs
    print(f"\n{'='*60}")
    print(f"PHASE 4: OPTIMIZING EPOCHS")
    print(f"{'='*60}")
    phase4_complete, saved_value, saved_iou = check_phase_completion(output_dir, 'epochs')
    if phase4_complete:
        print(f"⏭️  PHASE 4 ALREADY COMPLETE - Skipping")
        print(f"   Loaded best epochs: {saved_value} (Final IoU: {saved_iou:.4f})")
        best_params['epochs'] = saved_value
        phase4_data = []
        # Ambil best val_iou dari epochs.csv untuk best epochs
        import pandas as pd
        epochs_csv = os.path.join(output_dir, 'epochs.csv')
        df_epochs = pd.read_csv(epochs_csv)
        best_epochs_subset = df_epochs[df_epochs['param_value'] == saved_value]
        best_val_iou = best_epochs_subset['val_iou'].max() if not best_epochs_subset.empty else saved_iou
        print(f"   Best Val IoU during training: {best_val_iou:.4f}")
    else:
        best_params['epochs'], phase4_data = optimize_parameter(
            'epochs', param_space['epochs'], best_params, base_cfg, device, output_dir, model_type, is_epoch_search=True
        )
        # Ambil best val_iou dari param_value yang terpilih (best_params['epochs'])
        # Cari max val_iou dari semua epoch dalam konfigurasi best_params['epochs']
        best_epochs_data = [row for row in phase4_data if row['param_value'] == best_params['epochs']]
        best_val_iou = max(row['val_iou'] for row in best_epochs_data) if best_epochs_data else 0.0
    
    # Ringkasan akhir
    print(f"\n{'='*60}")
    print("OPTIMASI SELESAI!")
    print(f"{'='*60}")
    print(f"Parameter terbaik: {best_params}")
    print(f"Validation IoU terbaik: {best_val_iou:.4f}")
    
    # Simpan konfigurasi akhir dengan semua parameter lengkap
    final_config_path = os.path.join(output_dir, 'best_config.yaml')
    final_cfg = {
        # Model config
        'model': model_type,
        'encoder': encoder_type,
        'base_c': base_cfg.get('base_c', 32),
        
        # Best hyperparameters from OFAT
        'optimizer': best_params['optimizer'],
        'batch_size': best_params['batch_size'],
        'lr': best_params['lr'],
        'epochs': best_params['epochs'],
        
        # Loss configuration
        'loss_type': base_cfg.get('loss_type', 'bce_dice'),
        'focal_alpha': base_cfg.get('focal_alpha', 0.25),
        'focal_gamma': base_cfg.get('focal_gamma', 2.0),
        'bce_dice_weight': base_cfg.get('bce_dice_weight', 0.5),
        
        # Scheduler configuration
        'use_scheduler': base_cfg.get('use_scheduler', False),
        'scheduler_type': base_cfg.get('scheduler_type', 'cosine'),
        'warmup_epochs': base_cfg.get('warmup_epochs', 5),
        
        # Other training config
        'accum_steps': base_cfg.get('accum_steps', 1),
        'use_amp': base_cfg.get('use_amp', True),
        
        # Performance
        'best_val_iou': round(best_val_iou, 4),
        
        # Paths
        'split_json': base_cfg.get('split_json'),
        'checkpoint_dir': base_cfg.get('checkpoint_dir'),
        'save_name': base_cfg.get('save_name'),
        'metrics_out': base_cfg.get('metrics_out')
    }
    
    with open(final_config_path, 'w') as f:
        yaml.dump(final_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Konfigurasi terbaik disimpan ke: {final_config_path}")
    
    # Model terbaik sudah tersimpan dari fase 4 (epoch search)
    checkpoint_dir = base_cfg.get('checkpoint_dir', 'Save_models')
    final_model_path = os.path.join(checkpoint_dir, base_cfg.get('save_name', 'best_model.pth'))
    
    print(f"\n{'='*60}")
    print("OFAT COMPLETE")
    print(f"{'='*60}")
    print(f"Best model tersimpan: {final_model_path}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"{'='*60}")
    

    # === FAILURE CASE EXTRACTION (FP/FN) ===
    print(f"\n{'='*60}")
    print("Extracting failure cases (FP/FN) from test set...")
    print(f"{'='*60}")
    from PIL import Image
    import numpy as np

    # Load best model
    model = UNet(in_ch=3, n_classes=1, base_c=base_cfg.get('base_c',32), encoder=encoder_type)
    model_path = os.path.join(checkpoint_dir, base_cfg.get('save_name', 'best_model.pth'))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device); model.eval()

    # Load test set
    test_ds = FloodDataset(base_cfg['split_json'], split='test', use_preprocessed=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    fp_examples = []
    fn_examples = []
    for img, mask, idd in test_loader:
        img = img.to(device); mask = mask.to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(img)).cpu().numpy()[0,0]
        pred_bin = (pred > 0.5).astype('uint8')
        gt_arr = mask.cpu().numpy()[0,0].astype('uint8')
        # FP: pred=1, gt=0; FN: pred=0, gt=1
        fp_pix = np.sum((pred_bin==1) & (gt_arr==0))
        fn_pix = np.sum((pred_bin==0) & (gt_arr==1))
        if fp_pix > 0:
            fp_examples.append((img.cpu(), gt_arr, pred_bin, idd[0]))
        if fn_pix > 0:
            fn_examples.append((img.cpu(), gt_arr, pred_bin, idd[0]))
        if len(fp_examples) >= 3 and len(fn_examples) >= 3:
            break

    # Save error maps for FP and FN examples
    fail_dir = os.path.join(output_dir, 'failure_cases')
    os.makedirs(fail_dir, exist_ok=True)
    def save_error_map(img_tensor, gt, pred, case_type, id_str):
        img_np = img_tensor.squeeze().permute(1,2,0).numpy()
        img_disp = (img_np * 255).astype(np.uint8)
        err = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        FN = (gt == 1) & (pred == 0)
        FP = (gt == 0) & (pred == 1)
        err[FN] = [255, 0, 0]  # Red: missed flood
        err[FP] = [0, 255, 0]  # Green: false alarm
        # Overlay error map on image
        overlay = img_disp.copy()
        alpha = 0.5
        overlay[FN] = (overlay[FN] * (1-alpha) + np.array([255,0,0])*alpha).astype(np.uint8)
        overlay[FP] = (overlay[FP] * (1-alpha) + np.array([0,255,0])*alpha).astype(np.uint8)
        out_img = Image.fromarray(overlay)
        out_path = os.path.join(fail_dir, f'{case_type}_{id_str}.png')
        out_img.save(out_path)
        print(f"Saved {case_type} example: {out_path}")

    for img, gt, pred, id_str in fp_examples[:3]:
        save_error_map(img, gt, pred, 'FP', id_str)
    for img, gt, pred, id_str in fn_examples[:3]:
        save_error_map(img, gt, pred, 'FN', id_str)

    print(f"✓ Failure case examples saved to: {fail_dir}")

    return best_params, best_val_iou


def optimize_parameter(param_name, param_values, current_best, base_cfg, device, output_dir, model_type, is_epoch_search=False):
    """
    Mengoptimasi satu parameter dan menyimpan hasil ke CSV.
    Untuk epoch search (fase 4), akan save model terbaik.
    Support resume dari CSV yang sudah ada.
    """
    print(f"Menguji {len(param_values)} nilai untuk {param_name}")
    print(f"Parameter terbaik saat ini: {current_best}")
    print(f"Nilai yang akan diuji: {param_values}\n")
    
    best_value = current_best[param_name]
    best_iou = 0.0
    best_dice = 0.0
    best_score = 0.0  # gabungan iou+dice
    best_model_state = None
    all_results = []

    # Check jika CSV sudah ada - resume dari sana
    csv_filename = f"{param_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    tested_values = set()
    
    if os.path.exists(csv_path):
        print(f"⚠ Ditemukan CSV checkpoint: {csv_path}")
        import pandas as pd
        df = pd.read_csv(csv_path)
        if not df.empty:
            # Load semua hasil yang sudah ada
            all_results = df.to_dict('records')
            tested_values = set(df['param_value'].unique())
            
            # Cari best value dari hasil yang sudah ada - gunakan FINAL epoch (konsisten dengan logic baru)
            final_results = []
            for param_val in df['param_value'].unique():
                subset = df[df['param_value'] == param_val]
                # Ambil row dengan epoch tertinggi (final epoch)
                final_row = subset.loc[subset['epoch'].idxmax()]
                final_iou = final_row['val_iou']
                final_dice = final_row['val_dice']
                final_score = (final_iou + final_dice) / 2
                final_results.append({
                    'param_value': param_val,
                    'val_iou': final_iou,
                    'val_dice': final_dice,
                    'score': final_score
                })
            
            # Pilih param_value dengan score tertinggi
            best_result = max(final_results, key=lambda x: x['score'])
            best_value = best_result['param_value']
            
            # Convert numpy types ke Python native types
            if hasattr(best_value, 'item'):  # numpy scalar
                best_value = best_value.item()
            elif param_name == 'optimizer':  # string
                best_value = str(best_value)
            elif param_name in ['batch_size', 'epochs']:  # integer
                best_value = int(best_value)
            elif param_name == 'lr':  # float
                best_value = float(best_value)
            
            best_iou = float(best_result['val_iou'])
            best_dice = float(best_result['val_dice'])
            best_score = float(best_result['score'])
            
            print(f"✓ Loaded {len(tested_values)} tested values dari checkpoint")
            print(f"✓ Best dari checkpoint: {param_name}={best_value}, IoU={best_iou:.4f}, Dice={best_dice:.4f}, Score={best_score:.4f}")
            
            # Filter param_values untuk skip yang sudah di-test
            remaining_values = [v for v in param_values if v not in tested_values]
            if not remaining_values:
                print(f"✓ Semua nilai sudah di-test, skip parameter {param_name}")
                return best_value, all_results  # Return all_results, bukan best_iou
            print(f"✓ Sisa {len(remaining_values)} nilai yang belum di-test: {remaining_values}\n")
            param_values = remaining_values

    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend untuk multithreading
    import matplotlib.pyplot as plt

    for value in param_values:
        print(f"\n{'─'*60}")
        print(f"Testing {param_name} = {value}")

        # Create test params
        test_params = current_best.copy()
        test_params[param_name] = value
        
        # Convert semua values ke Python native types
        test_params['optimizer'] = str(test_params['optimizer'])
        test_params['batch_size'] = int(test_params['batch_size']) if hasattr(test_params['batch_size'], 'item') else int(test_params['batch_size'])
        test_params['lr'] = float(test_params['lr']) if hasattr(test_params['lr'], 'item') else float(test_params['lr'])
        test_params['epochs'] = int(test_params['epochs']) if hasattr(test_params['epochs'], 'item') else int(test_params['epochs'])

        # Use 10 epochs for search, unless this IS the epoch search
        eval_epochs = value if is_epoch_search else 10

        print(f"Config: optimizer={test_params['optimizer']}, batch_size={test_params['batch_size']}, lr={test_params['lr']}, epochs={eval_epochs}")

        # Train and collect per-epoch data
        epoch_data, final_iou, final_dice, model_state = train_with_params_detailed(
            test_params, base_cfg, device, eval_epochs, model_type, 
            save_model=is_epoch_search
        )

        # Store results with loss config and scheduler info
        for epoch_info in epoch_data:
            all_results.append({
                'model': model_type,
                'encoder': base_cfg.get('encoder', 'baseline'),
                'base_c': base_cfg.get('base_c', 32),
                'optimizer': test_params['optimizer'],
                'batch_size': test_params['batch_size'],
                'lr': test_params['lr'],
                'loss_type': base_cfg.get('loss_type', 'bce_dice'),
                'focal_alpha': base_cfg.get('focal_alpha', 0.25),
                'focal_gamma': base_cfg.get('focal_gamma', 2.0),
                'use_scheduler': base_cfg.get('use_scheduler', False),
                'scheduler_type': base_cfg.get('scheduler_type', 'none'),
                'warmup_epochs': base_cfg.get('warmup_epochs', 0),
                'target_param': param_name,
                'param_value': value,
                'epoch': epoch_info['epoch'],
                'train_loss': epoch_info['train_loss'],
                'val_iou': epoch_info['val_iou'],
                'val_dice': epoch_info['val_dice']
            })

        print(f"→ Val IoU Akhir: {final_iou:.4f}, Val Dice Akhir: {final_dice:.4f}")

        # Simpan kurva train/val IoU dan Dice
        epochs = [e['epoch'] for e in epoch_data]
        val_ious = [e['val_iou'] for e in epoch_data]
        val_dices = [e['val_dice'] for e in epoch_data]
        train_losses = [e['train_loss'] for e in epoch_data]

        plt.figure(figsize=(10,5))
        plt.plot(epochs, val_ious, label='Val IoU', marker='o')
        plt.plot(epochs, val_dices, label='Val Dice', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'{param_name}={value} | Val IoU & Dice')
        plt.legend()
        plt.grid(True)
        curve_path = os.path.join(output_dir, f'{param_name}_{value}_curve.png')
        plt.savefig(curve_path)
        plt.close()
        print(f"✓ Kurva Val IoU/Dice disimpan: {curve_path}")

        # Pilih berdasarkan rata-rata IoU+Dice (konsisten dengan train.py)
        final_score = (final_iou + final_dice) / 2
        if final_score > best_score:
            best_score = final_score
            best_iou = final_iou
            best_dice = final_dice
            best_value = value
            # Save model state untuk fase 4 (epoch search)
            if is_epoch_search and model_state is not None:
                best_model_state = model_state
            print(f"✓ NILAI TERBAIK BARU {param_name}={value}, IoU={final_iou:.4f}, Dice={final_dice:.4f}, Score={final_score:.4f}")

    # Simpan ke CSV dengan semua parameter configuration
    csv_filename = f"{param_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    fieldnames = ['encoder',
        'model', 'base_c', 'optimizer', 'batch_size', 'lr', 
        'loss_type', 'focal_alpha', 'focal_gamma',
        'use_scheduler', 'scheduler_type', 'warmup_epochs',
        'target_param', 'param_value', 'epoch', 'train_loss', 'val_iou', 'val_dice'
    ]

    # Overwrite CSV dengan hasil lengkap (checkpoint lama + hasil baru)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✓ Hasil disimpan ke: {csv_path}")
    print(f"✓ {param_name} terbaik: {best_value} (IoU={best_iou:.4f}, Dice={best_dice:.4f})")

    # Save model jika ini fase epoch search (fase 4)
    if is_epoch_search and best_model_state is not None:
        checkpoint_dir = base_cfg.get('checkpoint_dir', 'Save_models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, base_cfg.get('save_name', 'best_model.pth'))
        torch.save(best_model_state, model_path)
        print(f"✓ Best model saved: {model_path} (IoU={best_iou:.4f}, Dice={best_dice:.4f})")

    # Convert best_value ke Python native type sebelum return
    if hasattr(best_value, 'item'):
        best_value = best_value.item()
    elif param_name == 'optimizer':
        best_value = str(best_value)
    elif param_name in ['batch_size', 'epochs']:
        best_value = int(best_value)
    elif param_name == 'lr':
        best_value = float(best_value)
    
    return best_value, all_results


def train_with_params_detailed(params, base_cfg, device, eval_epochs, model_type, save_model=False):
    """
    Melatih model dan mengembalikan data per-epoch + IoU akhir.
    Returns: (epoch_data_list, best_val_iou, best_model_state)
    """
    from torch.optim import SGD, Adam

    # Load datasets - train pakai preprocessed, val/test pakai original
    train_ds = FloodDataset(base_cfg['split_json'], split='train', use_preprocessed=True)
    val_ds = FloodDataset(base_cfg['split_json'], split='val', use_preprocessed=False)

    print(f"  Dataset: Train={len(train_ds)}, Val={len(val_ds)}")   

    train_loader = DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"  Batches: Train={len(train_loader)}, Val={len(val_loader)}")

    # Build model berdasarkan encoder
    base_c = base_cfg.get('base_c', 32)
    encoder = base_cfg.get('encoder', 'baseline')
    model = UNet(in_ch=3, n_classes=1, base_c=base_c, encoder=encoder).to(device)

    # Set model name
    encoder_names = {
        'baseline': 'Baseline U-Net',
        'resnet50': 'U-Net+ResNet50',
        'efficientnet_b1': 'U-Net+EfficientNet-B1'
    }
    model_name = encoder_names.get(encoder, encoder)

    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name} (base_c={base_c})")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Pilih optimizer dengan weight decay
    opt_name = params['optimizer'].lower()
    if opt_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=1e-4)
    elif opt_name == 'adam':
        optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    else:  # adamw
        optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-2)

    # Learning rate scheduler
    scheduler = None
    if base_cfg.get('use_scheduler', False):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        warmup_epochs = base_cfg.get('warmup_epochs', 5)
        scheduler = CosineAnnealingLR(optimizer, T_max=eval_epochs-warmup_epochs, eta_min=params['lr']*0.01)

    # Setup AMP scaler
    scaler = amp.GradScaler() if torch.cuda.is_available() else None

    # Loss configuration
    loss_fn = base_cfg.get('loss_type', 'bce_dice')
    loss_cfg = {
        'focal_alpha': base_cfg.get('focal_alpha', 0.25),
        'focal_gamma': base_cfg.get('focal_gamma', 2.0),
        'bce_dice_weight': base_cfg.get('bce_dice_weight', 0.5)
    }

    # Training loop dengan logging per epoch
    epoch_data = []
    best_val_iou = 0.0
    best_val_dice = 0.0
    best_val_score = 0.0  # gabungan iou+dice
    best_model_state = None
    warmup_epochs = base_cfg.get('warmup_epochs', 5) if scheduler else 0

    for epoch in range(1, eval_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler,
                                    loss_fn=loss_fn, loss_cfg=loss_cfg)
        val_metrics = compute_metrics(model, val_loader, device)
        val_iou = val_metrics['iou']
        val_dice = val_metrics['dice']

        # Learning rate scheduling with warmup
        if scheduler:
            if epoch > warmup_epochs:
                scheduler.step()
            else:
                # Linear warmup
                warmup_lr = params['lr'] * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

        epoch_data.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_iou': round(val_iou, 4),
            'val_dice': round(val_dice, 4)
        })

        val_score = (val_iou + val_dice) / 2
        if val_score > best_val_score:
            best_val_iou = val_iou
            best_val_dice = val_dice
            best_val_score = val_score
            if save_model:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch}/{eval_epochs}: loss={train_loss:.4f}, val_iou={val_iou:.4f}, val_dice={val_dice:.4f}, score={val_score:.4f} (terbaik: {best_val_score:.4f})")

    return epoch_data, best_val_iou, best_val_dice, best_model_state



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimasi OFAT untuk U-Net models')
    parser.add_argument('--config', default='src/experiments/Config/unet.yaml', help='File konfigurasi dasar')
    parser.add_argument('--output', default='Results/OFAT', help='Direktori output')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MEMULAI OFAT OPTIMIZATION")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print("="*60)
    
    best_params, best_iou = OFAT_optimize(args.config, args.output)
    
    print(f"\n{'='*60}")
    print("SEMUA OPTIMASI SELESAI!")
    print(f"{'='*60}")
    print(f"Parameter terbaik: {best_params}")
    print(f"Validation IoU terbaik: {best_iou:.4f}")
    print(f"\nHasil tersimpan di: {args.output}/")
    print(f"  - optimizer.csv")
    print(f"  - batch_size.csv")
    print(f"  - lr.csv")
    print(f"  - epochs.csv")
    print(f"  - best_config.yaml")
    print("="*60)
