# U-Net Flood Segmentation with OFAT Hyperparameter Optimization

Deep learning approach for flood segmentation using U-Net architecture with various encoders (Baseline, ResNet50, EfficientNet-B1) and One-Factor-At-a-Time (OFAT) hyperparameter optimization.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Experiment Setup](#experiment-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)

## 🎯 Overview

This project implements flood segmentation from satellite imagery using semantic segmentation with U-Net architecture. The main features include:

- **3 Model Variants**: Baseline U-Net, U-Net + ResNet50, U-Net + EfficientNet-B1
- **OFAT Optimization**: Systematic hyperparameter tuning (optimizer, batch size, learning rate, epochs)
- **Comprehensive Training**: Focal-Dice loss, Cosine scheduler with warmup, AMP support
- **Multiple Seeds**: Reproducibility testing with 3 random seeds (0, 42, 123)
- **Statistical Analysis**: Mean ± Std metrics across multiple runs

## 📊 Dataset

- **Source**: Flood area segmentation from satellite imagery
- **Total Images**: 480 images
- **Split Ratio**: 60% Train (288), 20% Val (96), 20% Test (96)
- **Location**: `Data/raw/`
- **Preprocessing**: Automatic resize and normalization to `Data/processed/`

### Data Structure
```
Data/
├── raw/
│   ├── Image/          # RGB satellite images
│   ├── Mask/           # Binary flood masks
│   └── metadata.csv    # Dataset metadata
├── processed/          # Preprocessed data (auto-generated)
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── comparison/
│   ├── val/
│   └── test/
└── splits/
    └── splits.json     # Train/val/test split indices (fixed)
```

## 🏗️ Model Architecture

### 1. Baseline U-Net
- Standard U-Net with configurable base channels
- No pretrained encoder
- Lightweight and fast

### 2. U-Net + ResNet50
- U-Net with ResNet50 encoder (pretrained on ImageNet)
- Better feature extraction
- Higher accuracy

### 3. U-Net + EfficientNet-B1
- U-Net with EfficientNet-B1 encoder (pretrained on ImageNet)
- Efficient and accurate
- Compound scaling

## 🔬 Experiment Setup

### OFAT (One-Factor-At-a-Time) Optimization

Sequential hyperparameter optimization approach:

1. **Phase 1 - Optimizer**: `adamw`, `adam`, `sgd`
2. **Phase 2 - Batch Size**: `2`, `4`, `8`
3. **Phase 3 - Learning Rate**: `1e-5`, `3e-5`, `5e-5`, `8e-5`, `1e-4`, `1.5e-4`, `2e-4`, `3e-4`, `5e-4`
4. **Phase 4 - Epochs**: `40`, `60`, `80`, `100`, `120`, `150`, `200`

**Process**: Each phase locks the best parameter from previous phases and tests the next parameter. After all phases complete, `best_config.yaml` contains optimal hyperparameters.

### Seed Rerun Experiments

After finding best config via OFAT, models are retrained with 3 different seeds:
- **Seeds**: `0`, `42`, `123`
- **Purpose**: Measure model stability and report mean ± std
- **Configs Tested**: 
  - `best_config.yaml` (optimal from OFAT)
  - `random_config.yaml` (suboptimal baseline for comparison)

### Training Configuration

- **Loss**: Focal-Dice (α=0.25, γ=2.0, weight=0.5)
- **Scheduler**: Cosine annealing with warmup (5 epochs)
- **AMP**: Mixed precision training enabled
- **Metrics**: IoU (Intersection over Union), Dice Coefficient, Accuracy, Precision, Recall, F1


## 🚀 Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended: ≥6GB VRAM)
- PyTorch 1.12+

### Setup

```bash
# Clone repository
git clone https://github.com/adka-fawaid/FloodSegmentationUsingU-Net-Resnet50.git
cd FloodSegmentationUsingU-Net-Resnet50

# Create conda environment
conda create -n U-net python=3.9
conda activate U-net

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.0.0
pyyaml>=5.4.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
albumentations>=1.0.0
```

## 📖 Usage

### 1. Data Preprocessing

```bash
# Create train/val/test splits
python src/preprocessing/create_splits.py

# Preprocess images
python src/preprocessing/preprocess.py
```

### 2. OFAT Hyperparameter Optimization

Run OFAT for each model (recommended: sequential to avoid GPU OOM):

```bash
# Baseline U-Net
python main.py action:ofat_baseline

# U-Net + ResNet50
python main.py action:ofat_resnet50

# U-Net + EfficientNet-B1
python main.py action:ofat_efficientnet
```

**Output**: 
- `Results/OFAT_<model>/best_config.yaml` - Best hyperparameters
- `Results/OFAT_<model>/*.csv` - Detailed metrics per phase
- `Save_models/best_unet_<model>.pth` - Best trained model

### 3. Training with Best Config

```bash
# Train Baseline
python main.py action:train_baseline

# Train ResNet50
python main.py action:train_resnet50

# Train EfficientNet
python main.py action:train_efficientnet
```

### 4. Multiple Seed Runs (Reproducibility)

**ResNet50 - Best Config:**
```bash
python src/training/rerun_resnet50_seeds.py \
    --config Results/OFAT_resnet50/best_config.yaml \
    --output_dir Results/ResNet50_rerun_seeds
```

**ResNet50 - Random Config:**
```bash
python src/training/rerun_resnet50_seeds.py \
    --config src/experiments/Config/random_config.yaml \
    --output_dir Results/ResNet50_rerun_seeds
```

**EfficientNet B1 - Random Config:**
```bash
python src/training/rerun_resnet50_seeds.py --run_efficientnet_random
```

**Output Structure**:
```
Results/<Model>_rerun_seeds/
├── best_config/           # Results for best_config.yaml
│   ├── seed_0/
│   │   ├── training_log.csv
│   │   ├── metrics_summary.csv
│   │   └── best_model.pth
│   ├── seed_42/
│   ├── seed_123/
│   ├── rerun_seeds_summary.csv
│   └── seed_statistics.csv    # Mean ± Std
└── random_config/         # Results for random_config.yaml
    └── (same structure)
```

### 5. Inference & Visualization

```bash
# Visualize results (comparison of all models)
python src/inference/visualize_results.py \
    --config_baseline Results/OFAT_baseline/best_config.yaml \
    --config_resnet50 Results/OFAT_resnet50/best_config.yaml \
    --config_efficientnet Results/OFAT_efficientnet/best_config.yaml \
    --output_dir Results/visualizations

# Single model inference
python src/inference/infer.py \
    --config Results/OFAT_resnet50/best_config.yaml \
    --checkpoint Save_models/best_unet_resnet50.pth \
    --image_path Data/raw/Image/sample.jpg
```

## 📁 Project Structure

```
UNET+FBRM/
├── Data/
│   ├── raw/                    # Original dataset
│   │   ├── Image/              # RGB satellite images (480 total)
│   │   ├── Mask/               # Binary flood masks
│   │   └── metadata.csv
│   ├── processed/              # Preprocessed data (auto-generated)
│   │   ├── train/              # 288 images
│   │   ├── val/                # 96 images
│   │   └── test/               # 96 images
│   └── splits/
│       └── splits.json         # Fixed train/val/test split
├── Results/
│   ├── OFAT_baseline/          # OFAT results for baseline
│   │   ├── best_config.yaml
│   │   ├── optimizer.csv
│   │   ├── batch_size.csv
│   │   ├── lr.csv
│   │   ├── epochs.csv
│   │   └── search_space_summary.csv
│   ├── OFAT_resnet50/          # OFAT results for ResNet50
│   ├── OFAT_efficientnet/      # OFAT results for EfficientNet B1
│   ├── ResNet50_rerun_seeds/   # Multi-seed runs (ResNet50)
│   │   ├── best_config/
│   │   │   ├── seed_0/
│   │   │   ├── seed_42/
│   │   │   ├── seed_123/
│   │   │   ├── rerun_seeds_summary.csv
│   │   │   └── seed_statistics.csv
│   │   └── random_config/
│   ├── EfficientNet_rerun_seeds/  # Multi-seed runs (EfficientNet)
│   └── comprehensive_metrics.csv   # All models final test metrics
├── Save_models/                # Trained model checkpoints
│   ├── best_unet_baseline.pth
│   ├── best_unet_resnet50.pth
│   └── best_unet_efficientnet_b1.pth
├── src/
│   ├── Dataset/
│   │   ├── loader.py          # FloodDataset class
│   │   └── transform.py       # Data augmentation (training only)
│   ├── models/
│   │   └── unet.py            # U-Net with multiple encoder options
│   ├── training/
│   │   ├── engine.py          # Training/validation loops
│   │   ├── losses.py          # Focal-Dice, BCE-Dice losses
│   │   ├── metrics.py         # IoU, Dice, Accuracy, etc.
│   │   ├── train.py           # Main training script
│   │   └── rerun_resnet50_seeds.py  # Multi-seed runner
│   ├── experiments/
│   │   ├── OFAT.py            # OFAT optimization script
│   │   └── Config/            # Model configurations
│   │       ├── unet.yaml
│   │       ├── unet-resnet.yaml
│   │       ├── unet_efficientnet.yaml
│   │       └── random_config.yaml
│   ├── inference/
│   │   ├── infer.py           # Single image inference
│   │   └── visualize_results.py  # Multi-model comparison
│   ├── preprocessing/
│   │   ├── preprocess.py      # Image preprocessing pipeline
│   │   └── create_splits.py   # Dataset splitting (60/20/20)
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── viz.py             # Visualization tools
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md
```

## 📈 Results

### Best Configurations (from OFAT)

**ResNet50:**
- Optimizer: AdamW
- Batch Size: 8
- Learning Rate: 0.0003
- Epochs: 100
- **Best Val IoU**: 0.7717 ± 0.0089
- **Test IoU**: 0.7664 ± 0.0064
- **Test Dice**: 0.8499 ± 0.0050

**EfficientNet B1:**
- Optimizer: Adam
- Batch Size: 4
- Learning Rate: 0.0002
- Epochs: 80
- **Best Val IoU**: 0.7572 ± 0.0056
- **Test IoU**: 0.7619 ± 0.0065
- **Test Dice**: 0.8452 ± 0.0062

### Output Files

1. **Best Config Files**: `Results/OFAT_<model>/best_config.yaml`
   - Optimal hyperparameters for each model
   - Best validation IoU achieved during OFAT

2. **Training Logs**: 
   - `training_log.csv` - Per-epoch metrics (train_loss, val_iou, val_dice, val_accuracy, etc.)
   - `metrics_summary.csv` - Final test set evaluation
   - `seed_statistics.csv` - Mean ± Std across 3 seeds

3. **Model Checkpoints**: `Save_models/best_unet_<model>.pth`
   - Best model based on validation IoU
   - Saved during training at best epoch

4. **OFAT Analysis**:
   - `optimizer.csv` - Optimizer comparison (AdamW, Adam, SGD)
   - `batch_size.csv` - Batch size sweep (2, 4, 8)
   - `lr.csv` - Learning rate sweep (9 values)
   - `epochs.csv` - Epoch sweep (40-200)
   - `search_space_summary.csv` - Best value per parameter

### Folder Structure After Training

```
Results/
├── OFAT_resnet50/
│   ├── best_config.yaml           # Best hyperparameters
│   ├── optimizer.csv              # Phase 1: Optimizer results
│   ├── batch_size.csv             # Phase 2: Batch size results
│   ├── lr.csv                     # Phase 3: Learning rate results
│   ├── epochs.csv                 # Phase 4: Epochs results
│   ├── search_space_summary.csv   # Summary of all parameters
│   └── *_curve.png                # Training curves for each config
├── ResNet50_rerun_seeds/
│   ├── best_config/
│   │   ├── seed_0/
│   │   │   ├── training_log.csv
│   │   │   ├── metrics_summary.csv
│   │   │   └── best_model.pth
│   │   ├── seed_42/ (same files)
│   │   ├── seed_123/ (same files)
│   │   ├── rerun_seeds_summary.csv
│   │   └── seed_statistics.csv    # Mean: 0.7664, Std: 0.0064 (Test IoU)
│   └── random_config/ (same structure)
└── comprehensive_metrics.csv      # All models comparison
```

## 🔧 Configuration Files

### Model Configs (`src/experiments/Config/`)

- `unet.yaml` - Baseline U-Net configuration
- `unet-resnet.yaml` - U-Net + ResNet50 configuration
- `unet_efficientnet.yaml` - U-Net + EfficientNet-B1 configuration
- `random_config.yaml` - Suboptimal config for baseline comparison

### Key Parameters (Example: ResNet50 Best Config)

```yaml
model: unet_resnet50
encoder: resnet50
base_c: 96                    # Base channels
optimizer: adamw              # From OFAT Phase 1
batch_size: 8                 # From OFAT Phase 2
lr: 0.0003                    # From OFAT Phase 3
epochs: 100                   # From OFAT Phase 4
loss_type: focal_dice
focal_alpha: 0.25
focal_gamma: 2.0
bce_dice_weight: 0.5
use_scheduler: true
scheduler_type: cosine
warmup_epochs: 5
use_amp: true                 # Mixed precision training
accum_steps: 1                # Gradient accumulation
```

##  Troubleshooting

### GPU Out of Memory
- Reduce `batch_size` in config (try `batch_size: 2`)
- Run models sequentially (not parallel)
- Disable AMP: `use_amp: false`
- Use gradient accumulation: `accum_steps: 2` or higher

### Training Too Slow
- Enable AMP: `use_amp: true` (default)
- Increase `batch_size` if GPU memory allows
- Use `accum_steps` for effective larger batch size
- Ensure CUDA is properly installed

### Inconsistent Results
- Check random seed settings in code
- Verify data preprocessing is deterministic
- Run multiple seed experiments (provided scripts handle this)
- Check if data augmentation is disabled for validation/test

### OFAT Script Errors
- Ensure all dependencies are installed
- Check file paths in config files
- Verify `Data/splits/splits.json` exists
- Run preprocessing first: `python src/preprocessing/create_splits.py`

### YAML Config Corruption
- If `best_config.yaml` contains binary data, regenerate using OFAT script
- Script automatically converts numpy types to Python native types
- Do not manually edit generated YAML files

## 📝 Notes

- **OFAT Duration**: ~4-8 hours per model (GPU-dependent)
- **GPU Requirements**: Minimum 6GB VRAM (8GB+ recommended)
- **Checkpoint Saving**: Incremental (can resume if interrupted)
- **Analysis Scripts**: Temporary scripts are gitignored
- **Reproducibility**: Fixed seed (42) for OFAT, multiple seeds for final evaluation
- **Data Augmentation**: Only applied to training set, not validation/test

## 📝 Notes

- **OFAT Duration**: ~4-8 hours per model (GPU-dependent)
- **GPU Requirements**: Minimum 6GB VRAM (8GB+ recommended)
- **Checkpoint Saving**: Incremental (can resume if interrupted)
- **Analysis Scripts**: Temporary scripts are gitignored
- **Reproducibility**: Fixed seed (42) for OFAT, multiple seeds for final evaluation
- **Data Augmentation**: Only applied to training set, not validation/test

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{unet_flood_ofat2026,
  author = {Your Name},
  title = {U-Net Flood Segmentation with OFAT Hyperparameter Optimization},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/adka-fawaid/FloodSegmentationUsingU-Net-Resnet50}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Author

**Adka Fawaid**
- GitHub: [@adka-fawaid](https://github.com/adka-fawaid)
- Repository: [FloodSegmentationUsingU-Net-Resnet50](https://github.com/adka-fawaid/FloodSegmentationUsingU-Net-Resnet50)

## 📚 References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
