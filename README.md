# U-Net Flood Detection with OFAT Hyperparameter Optimization

Deep learning approach for flood detection using U-Net architecture with various encoders (Baseline, ResNet50, EfficientNet-B1) and One-Factor-At-a-Time (OFAT) hyperparameter optimization.

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

This project implements flood detection using semantic segmentation with U-Net architecture. The main features include:

- **3 Model Variants**: Baseline U-Net, U-Net + ResNet50, U-Net + EfficientNet-B1
- **OFAT Optimization**: Systematic hyperparameter tuning for each model
- **Comprehensive Training**: Focal-Dice loss, Cosine scheduler, AMP support
- **Multiple Seeds**: Reproducibility testing with different random seeds

## 📊 Dataset

- **Source**: Flood area segmentation dataset
- **Location**: `Data/raw/`
- **Split**: Train/Validation/Test (configurable via `Data/splits/splits.json`)
- **Preprocessing**: Automatic preprocessing pipeline to `Data/processed/`

### Data Structure
```
Data/
├── raw/
│   ├── Image/          # RGB images
│   ├── Mask/           # Binary masks
│   └── metadata.csv
├── processed/          # Preprocessed data (auto-generated)
└── splits/
    └── splits.json     # Train/val/test split indices
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

Sequential hyperparameter optimization:
1. **Phase 1**: Optimizer (`adamw`, `adam`, `sgd`)
2. **Phase 2**: Batch size (`2`, `4`, `8`)
3. **Phase 3**: Learning rate (`1e-5` to `5e-4`)
4. **Phase 4**: Epochs (`40`, `60`, `80`, `100`, `120`, `150`, `200`)

Each phase locks the best parameter from previous phases and tests the next parameter.

### Training Configuration

- **Loss**: Focal-Dice (α=0.25, γ=2.0)
- **Scheduler**: Cosine annealing with warmup (5 epochs)
- **AMP**: Mixed precision training enabled
- **Metrics**: IoU (Intersection over Union), Dice coefficient

## 🚀 Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended: ≥6GB VRAM)
- PyTorch 1.12+

### Setup

```bash
# Clone repository
git clone [<repository-url>](https://github.com/adka-fawaid/FloodSegmentationUsingU-Net-Resnet50.git)
cd FloodSegmentationUsingU-Net-Resnet50

# Create conda environment
conda create -n pytorch-gnn python=3.9
conda activate pytorch-gnn

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

```bash
# Run with 3 different seeds (0, 42, 123)
python src/training/rerun_resnet50_seeds.py \
    --config Results/OFAT_resnet50/best_config.yaml \
    --output_dir Results/ResNet50_rerun_seeds
```

**Output Structure**:
```
Results/ResNet50_rerun_seeds/
├── best_config/           # Results for best_config.yaml
│   ├── seed_0/
│   ├── seed_42/
│   ├── seed_123/
│   └── rerun_seeds_summary.csv
└── random_config/         # Results for random_config.yaml
    ├── seed_0/
    ├── seed_42/
    ├── seed_123/
    └── rerun_seeds_summary.csv
```

### 5. Inference & Visualization

```bash
# Visualize results (comparison of all models)
python src/inference/visualize_results.py \
    --config_baseline Results/OFAT_baseline/best_config.yaml \
    --config_resnet50 Results/OFAT_resnet50/best_config.yaml \
    --config_efficientnet Results/OFAT_efficientnet/best_config.yaml \
    --output_dir Results/visualizations
```

## 📁 Project Structure

```
UNET+FBRM/
├── Data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/val/test splits
├── Results/
│   ├── OFAT_baseline/          # OFAT results for baseline
│   ├── OFAT_resnet50/          # OFAT results for ResNet50
│   ├── OFAT_efficientnet/      # OFAT results for EfficientNet
│   └── ResNet50_rerun_seeds/   # Multiple seed runs
├── Save_models/                # Trained model checkpoints
├── src/
│   ├── Dataset/
│   │   ├── loader.py          # Dataset class
│   │   └── transform.py       # Data augmentation
│   ├── models/
│   │   └── unet.py            # U-Net implementation
│   ├── training/
│   │   ├── engine.py          # Training loop
│   │   ├── losses.py          # Loss functions
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── train.py           # Training script
│   │   └── rerun_resnet50_seeds.py  # Multi-seed runs
│   ├── experiments/
│   │   ├── OFAT.py            # OFAT optimization
│   │   └── Config/            # Model configurations
│   ├── inference/
│   │   ├── infer.py           # Inference script
│   │   └── visualize_results.py  # Visualization
│   ├── preprocessing/
│   │   ├── preprocess.py      # Preprocessing pipeline
│   │   └── create_splits.py   # Dataset splitting
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── viz.py             # Visualization tools
├── main.py                    # Main entry point
└── README.md
```

## 📈 Results

### Expected Outputs

After running OFAT and training:

1. **Best Config Files**: `Results/OFAT_<model>/best_config.yaml`
   - Optimal hyperparameters for each model
   - Best validation IoU achieved

2. **Training Logs**: 
   - `training_log.csv` - Per-epoch metrics
   - `metrics_summary.csv` - Final test metrics

3. **Model Checkpoints**: `Save_models/best_unet_<model>.pth`
   - Best model based on validation IoU

4. **OFAT Analysis**:
   - `optimizer.csv`, `batch_size.csv`, `lr.csv`, `epochs.csv`
   - Detailed metrics for each tested hyperparameter

### Example Results Structure

```
Results/OFAT_resnet50/
├── best_config.yaml           # Best hyperparameters
├── optimizer.csv              # Phase 1 results
├── batch_size.csv             # Phase 2 results
├── lr.csv                     # Phase 3 results
├── epochs.csv                 # Phase 4 results
└── failure_cases/             # FP/FN examples
    ├── FP_*.png
    └── FN_*.png
```

## 🔧 Configuration Files

### Model Configs (`src/experiments/Config/`)

- `unet.yaml` - Baseline U-Net
- `unet-resnet.yaml` - U-Net + ResNet50
- `unet_efficientnet.yaml` - U-Net + EfficientNet-B1

### Key Parameters

```yaml
model: unet_resnet50
encoder: resnet50
base_c: 96
optimizer: adamw
batch_size: 8
lr: 0.0001
epochs: 40
loss_type: focal_dice
focal_alpha: 0.25
focal_gamma: 2.0
use_scheduler: true
scheduler_type: cosine
warmup_epochs: 5
use_amp: true
```

## 🐛 Troubleshooting

### GPU Out of Memory
- Reduce `batch_size` in config
- Run models sequentially (not parallel)
- Disable AMP: `use_amp: false`

### Training Too Slow
- Enable AMP: `use_amp: true`
- Increase `batch_size` if GPU memory allows
- Use `accum_steps` for gradient accumulation

### Inconsistent Results
- Check random seed settings
- Verify data preprocessing
- Run multiple seed experiments

## 📝 Notes

- OFAT optimization takes several hours per model
- Recommended to run on GPU (≥6GB VRAM)
- Results are saved incrementally (can resume if interrupted)
- All temporary analysis scripts are gitignored

## 📄 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]

## 📚 References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
