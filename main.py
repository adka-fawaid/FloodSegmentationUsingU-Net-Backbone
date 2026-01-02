#!/usr/bin/env python3
"""
Main script to run the complete pipeline or individual steps.
Compares 3 U-Net architectures:
1. Baseline U-Net
2. U-Net + ResNet50 encoder
3. U-Net + EfficientNet-B1 encoder

Usage:
    python main.py all                       # Run complete pipeline
    python main.py action:create_splits      # Run specific action
    python main.py action:preprocess
    python main.py action:train_baseline
    python main.py action:train_resnet50
    python main.py action:train_efficientnet
    python main.py action:visualize
"""
import sys, subprocess, os, json, glob

def run(cmd):
    print('\n' + '='*60)
    print('>', cmd)
    print('='*60)
    result = subprocess.call(cmd, shell=True)
    if result != 0:
        print(f'ERROR: Command failed with exit code {result}')
        sys.exit(result)
    return result

def check_splits_complete():
    """Cek apakah splits sudah dibuat"""
    splits_path = 'Data/splits/splits.json'
    if not os.path.exists(splits_path):
        return False
    try:
        with open(splits_path) as f:
            data = json.load(f)
            return 'train' in data and 'val' in data and 'test' in data
    except:
        return False

def check_preprocessing_complete():
    """Cek apakah preprocessing sudah selesai"""
    train_images = 'Data/processed/train/images'
    train_masks = 'Data/processed/train/masks'
    if not os.path.exists(train_images) or not os.path.exists(train_masks):
        return False
    # Cek ada minimal beberapa file
    img_count = len(glob.glob(os.path.join(train_images, '*')))
    mask_count = len(glob.glob(os.path.join(train_masks, '*')))
    return img_count > 0 and mask_count > 0

def check_training_complete(model_name, output_dir, model_path):
    """Cek apakah training model sudah selesai"""
    config_path = os.path.join(output_dir, 'best_config.yaml')
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return False
    return True

def check_visualization_complete():
    """Cek apakah visualization sudah selesai untuk semua 3 model"""
    vis_dirs = [
        'Results/unet_baseline',
        'Results/unet_resnet50',
        'Results/unet_efficientnet_b1'
    ]
    for vis_dir in vis_dirs:
        if not os.path.exists(vis_dir):
            return False
        # Cek ada file PNG
        png_files = glob.glob(os.path.join(vis_dir, '*.png'))
        if len(png_files) == 0:
            return False
    return True

def run_all():
    """Run complete pipeline from start to finish dengan auto-skip untuk step yang sudah selesai"""
    print('\n' + '='*60)
    print('STARTING COMPLETE PIPELINE - 3 MODEL COMPARISON')
    print('💾 Checkpoint feature enabled: akan skip step yang sudah selesai')
    print('='*60)
    
    # Step 1: Create splits
    print('\n[STEP 1/6] Creating stratified splits...')
    if check_splits_complete():
        print('⏭️  SKIPPED - Splits sudah ada di Data/splits/splits.json')
    else:
        run('python src/preprocessing/create_splits.py --images Data/raw/Image --masks Data/raw/Mask --out Data/splits/splits.json')
    
    # Step 2: Preprocess (train only)
    print('\n[STEP 2/6] Preprocessing TRAIN data...')
    if check_preprocessing_complete():
        print('⏭️  SKIPPED - Preprocessing sudah selesai di Data/processed/train/')
    else:
        run('python src/preprocessing/preprocess.py --split Data/splits/splits.json --out Data/processed --size 256')
    
    # Step 3: Optimize Baseline U-Net with OFAT
    print('\n[STEP 3/6] Optimizing Baseline U-Net with OFAT...')
    if check_training_complete('Baseline U-Net', 'Results/OFAT_baseline', 'Save_models/best_unet_baseline.pth'):
        print('⏭️  SKIPPED - Baseline U-Net training sudah selesai')
    else:
        run('python "src/experiments/ofat.py" --config src/experiments/Config/unet.yaml --output Results/OFAT_baseline')
    
    # Step 4: Optimize U-Net + ResNet50 with OFAT
    print('\n[STEP 4/6] Optimizing U-Net + ResNet50 with OFAT...')
    if check_training_complete('U-Net + ResNet50', 'Results/OFAT_resnet50', 'Save_models/best_unet_resnet50.pth'):
        print('⏭️  SKIPPED - U-Net + ResNet50 training sudah selesai')
    else:
        run('python "src/experiments/ofat.py" --config src/experiments/Config/unet-resnet.yaml --output Results/OFAT_resnet50')
    
    # Step 5: Optimize U-Net + EfficientNet-B1 with OFAT
    print('\n[STEP 5/6] Optimizing U-Net + EfficientNet-B1 with OFAT...')
    if check_training_complete('U-Net + EfficientNet-B1', 'Results/OFAT_efficientnet', 'Save_models/best_unet_efficientnet_b1.pth'):
        print('⏭️  SKIPPED - U-Net + EfficientNet-B1 training sudah selesai')
    else:
        run('python "src/experiments/ofat.py" --config src/experiments/Config/unet_efficientnet.yaml --output Results/OFAT_efficientnet')
    
    # Step 6: Visualize all 3 models
    print('\n[STEP 6/6] Generating visualizations for all 3 models...')
    if check_visualization_complete():
        print('⏭️  SKIPPED - Visualization sudah selesai untuk semua 3 model')
    else:
        run('python src/inference/visualize_results.py --config_baseline src/experiments/Config/unet.yaml --config_resnet50 src/experiments/Config/unet-resnet.yaml --config_efficientnet src/experiments/Config/unet_efficientnet.yaml')
    
    print('\n' + '='*60)
    print('✓ PIPELINE COMPLETE!')
    print('='*60)
    print('\nResults saved to:')
    print('  - Data/processed/train/                    (preprocessed data)')
    print('  - Results/OFAT_baseline/     (Baseline U-Net optimization)')
    print('  - Results/OFAT_resnet50/     (ResNet50 optimization)')
    print('  - Results/OFAT_efficientnet/ (EfficientNet-B1 optimization)')
    print('  - Save_models/best_unet_baseline.pth       (Baseline model)')
    print('  - Save_models/best_unet_resnet50.pth       (ResNet50 model)')
    print('  - Save_models/best_unet_efficientnet_b1.pth (EfficientNet model)')
    print('  - Results/unet_baseline/                   (Baseline visualizations)')
    print('  - Results/unet_resnet50/                   (ResNet50 visualizations)')
    print('  - Results/unet_efficientnet_b1/            (EfficientNet visualizations)')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python main.py all                     # Run complete pipeline')
        print('  python main.py action:<action>         # Run specific action')
        print('\nAvailable actions:')
        print('  - create_splits      : Create train/val/test splits')
        print('  - preprocess         : Preprocess train data')
        print('  - train_baseline     : Optimize Baseline U-Net with OFAT')
        print('  - train_resnet50     : Optimize U-Net + ResNet50 with OFAT')
        print('  - train_efficientnet : Optimize U-Net + EfficientNet-B1 with OFAT')
        print('  - visualize          : Generate visualizations for all 3 models')
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == 'all':
        run_all()
    elif arg.startswith('action:'):
        action = arg.split(':',1)[1]
        if action == 'create_splits':
            run('python src/preprocessing/create_splits.py --images Data/raw/Image --masks Data/raw/Mask --out Data/splits/splits.json')
        elif action == 'preprocess':
            run('python src/preprocessing/preprocess.py --split Data/splits/splits.json --out Data/processed --size 256')
        elif action == 'train_baseline':
            run('python "src/experiments/ofat.py" --config src/experiments/Config/unet.yaml --output Results/OFAT_baseline')
        elif action == 'train_resnet50':
            run('python "src/experiments/ofat.py" --config src/experiments/Config/unet-resnet.yaml --output Results/OFAT_resnet50')
        elif action == 'train_efficientnet':
            run('python "src/experiments/ofat.py" --config src/experiments/Config/unet_efficientnet.yaml --output Results/OFAT_efficientnet')
        elif action == 'visualize':
            run('python src/inference/visualize_results.py --config_baseline src/experiments/Config/unet.yaml --config_resnet50 src/experiments/Config/unet-resnet.yaml --config_efficientnet src/experiments/Config/unet_efficientnet.yaml')
        else:
            print('Unknown action:', action)
            sys.exit(1)
    else:
        print('Invalid argument. Use "all" or "action:<action_name>"')
        sys.exit(1)
