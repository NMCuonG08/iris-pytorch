# Iris Segmentation (PyTorch + SegFormer)

Professional deep learning project for iris segmentation using SegFormer model on UBIRIS V2 dataset.

## 📋 Complete Step-by-Step Guide

### Step 1: Clone/Setup Project
```powershell
Set-Location "e:/MachineLearning/Eris"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

**What gets installed:**
- `torch>=2.2.0` - PyTorch deep learning framework
- `torchvision>=0.17.0` - Image utilities
- `transformers>=4.40.0` - HuggingFace pretrained models (SegFormer backbone)
- `albumentations>=1.4.0` - Advanced image augmentation
- `opencv-python>=4.8.0` - Image processing
- `python-dotenv>=1.0.0` - Load environment variables from `.env`
- `kagglehub>=0.1.0` - Download datasets from Kaggle

### Step 3: Setup Kaggle Credentials (One-Time Only)

1. Get your Kaggle API key:
   - Go to: [https://www.kaggle.com/account/api](https://www.kaggle.com/account/api)
   - Click "Create New API Token" → saves `kaggle.json`
   - Open `kaggle.json` and copy **username** and **key**

2. Create `.env` file from template:
   ```powershell
   Copy-Item .env.example .env
   ```

3. Edit `.env` file and fill in your credentials:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_api_key_from_json
   ```

4. Verify credentials work:
   ```powershell
   python -c "from src.utils.config import load_config; print('✓ .env loaded successfully')"
   ```

### Step 4: Download UBIRIS V2 Dataset (~3GB)

**⚠️ Important Note:**  
The **proper way** to acquire UBIRIS V2 is through the official application form at [https://iris.di.ubi.pt/ubiris2.html](https://iris.di.ubi.pt/ubiris2.html). The authors request researchers fill out a form before granting access.

This project uses a mirrored version on Kaggle for convenience. If you prefer the official dataset:
1. Fill the application form at [https://iris.di.ubi.pt/ubiris2.html](https://iris.di.ubi.pt/ubiris2.html)
2. Download directly from their server
3. Place in `dataset/raw/`
4. Run `python scripts/prepare_dataset.py` (skip to Step 5)

**To use the Kaggle mirror (automated):**
```powershell
python scripts/download_ubiris.py
```

**What happens:**
- Script reads `KAGGLE_USERNAME` and `KAGGLE_KEY` from `.env`
- Downloads `chinmoyslg/ubirisv2` dataset from Kaggle using `kagglehub`
- Automatically extracts using password: `UBIRIS2_IEEETPAMI_101109_200966`
- Extracts to `dataset/raw/`
- Takes 5-15 minutes depending on internet speed

**Expected output:**
```
Downloaded to: E:\MachineLearning\Eris\dataset\raw\...
Extracting dataset\raw\ubiris2_1.zip to dataset\raw
Done. Raw dataset is ready at: dataset/raw
```

**Folder structure after download:**
```
dataset/raw/
├── images/           # Contains all iris images
└── masks/            # Contains all iris masks
```

### Step 5: Organize & Split Dataset
```powershell
python scripts/prepare_dataset.py --source-images dataset/raw/CLASSES_400_300_Part1 --source-masks dataset/raw/ubiris --copy
```

If your extracted archive has a different folder name, point `--source-images` and `--source-masks` to the actual extracted folders. The prep script now searches recursively and supports `.tiff` / `.tif` as well as `.jpg` / `.png`.

**What this script does:**

1. **Match pairs**: Finds all image-mask pairs with matching filenames
   - Example: `001.jpg` matches with `001.png`
   - Only processes valid pairs (skips orphaned files)

2. **Copy to target**: Copies matched images/masks to organized folders
   - Images → `dataset/images/`
   - Masks → `dataset/masks/`
   - (Uses `--copy` flag to copy; omit for move)

3. **Random split**: Shuffles all pairs and splits into train/val/test
   - Default: 80% train, 20% validation, 0% test
   - Randomness controlled by `--seed` flag
   - Can change with `--train-ratio` and `--val-ratio`

4. **Generate split files**: Creates readable split lists
   - `dataset/splits/train.txt` - one image stem per line
   - `dataset/splits/val.txt` - one image stem per line
   - `dataset/splits/test.txt` - one image stem per line

**Expected output:**
```
Matching images and masks...
Found 2000 valid pairs
Splitting: train=1600, val=400, test=0
Prepared 2000 pairs
Train/Val/Test: 1600/400/0
Output images: dataset/images
Output masks: dataset/masks
```

**Folder structure after preparation:**
```
dataset/
├── images/              # 2000 iris images
├── masks/               # 2000 iris masks
├── splits/
│   ├── train.txt        # 1600 stems
│   ├── val.txt          # 400 stems
│   └── test.txt         # 0 stems
└── raw/                 # (can delete after preparation)
    ├── images/
    └── masks/
```

### Step 6: (Optional) Customize Training Config
Edit `configs/train.yaml` to change hyperparameters:

```yaml
training:
  epochs: 2              # Change to 50+ for production training
  batch_size: 8          # Reduce if CUDA out of memory
  lr: 0.0001             # Learning rate
  device: cuda           # Use 'cpu' if GPU unavailable

data:
  input_size: [512, 512] # Image resize dimensions
  num_workers: 4         # DataLoader threads

model:
  backbone_name: nvidia/segformer-b0-finetuned-ade-512-512  # Can use b1, b2, b3 for bigger models
```

### Step 7: Start Training
```powershell
python train.py --config configs/train.yaml
```

**What happens during training:**

1. **Model initialization**: Loads SegFormer backbone from HuggingFace
2. **Data loading**: Reads images/masks from `dataset/images/` and `dataset/masks/`
3. **Augmentation**: Applies aggressive transforms to training set only
   - Rotation, shift, scale, flip, brightness/contrast changes
   - Blur (motion/gaussian/median) to simulate poor iris capture
   - Noise (gaussian/ISO) to simulate low sensor quality
   - Denoising to improve robustness
4. **Forward pass**: Generates segmentation masks (pixel-wise predictions)
5. **Loss computation**: Combined Dice (60%) + Focal (40%) Loss
6. **Backward pass**: Gradient computation and model weight updates
7. **Validation**: Per-epoch validation on val set (no augmentation)
8. **Checkpointing**: Saves best model when val metric improves

**Real-time console output:**
```
PD Started training...
Epoch 1/2
 100%|████████| 200/200 [05:23<00:00, 1.61s/it]
Epoch 1/2 summary: loss_train=4.34595, acc_train=6.05%, loss_val=4.07431, acc_val=9.52%, miou_train=3.09%, miou_val=4.98% (best: 9.52% @ epoch 1)

Epoch 2/2
 100%|████████| 200/200 [05:18<00:00, 1.59s/it]
Epoch 2/2 summary: loss_train=3.12345, acc_train=15.23%, loss_val=2.98765, acc_val=18.45%, miou_train=8.25%, miou_val=10.14% (best: 18.45% @ epoch 2)

Training complete!
```

**Log files generated:**
- `runs/iris-segformer/checkpoints/best.pt` - Best model checkpoint
- `runs/iris-segformer/checkpoints/latest.pt` - Latest epoch checkpoint
- `runs/iris-segformer/logs/trainer.log` - Detailed training log
- `runs/iris-segformer/epoch_summary.txt` - Per-epoch metrics (human-readable)
- `runs/iris-segformer/metrics.csv` - Structured metrics for plotting

**Example `epoch_summary.txt` content:**
```
02/04/2026 11:19:33 Epoch 1/2 summary: loss_train=4.34595, acc_train=6.05%, loss_val=4.07431, acc_val=9.52%, miou_train=3.09%, miou_val=4.98% (best: 9.52% @ epoch 1)
02/04/2026 11:19:52 Epoch 2/2 summary: loss_train=3.12345, acc_train=15.23%, loss_val=2.98765, acc_val=18.45%, miou_train=8.25%, miou_val=10.14% (best: 18.45% @ epoch 2)
```

### Step 8: Run Post-Training Visualizations

#### 8.1 Original / Ground Truth / Prediction Comparison
Generate a 3-panel figure that shows the raw image, the human mask, and the model prediction.
The comparison uses **red** for ground truth and **green** for the AI prediction, so overlapping pixels appear **yellow**.

```powershell
python scripts/visualize_prediction.py `
  --checkpoint runs/iris-segformer/checkpoints/best.pt `
  --stem 0001 `
  --output runs/iris-segformer/visualizations/sample_comparison.png
```

If you want to use explicit files instead of a stem:

```powershell
python scripts/visualize_prediction.py `
  --checkpoint runs/iris-segformer/checkpoints/best.pt `
  --image dataset/images/0001.jpg `
  --mask dataset/masks/0001.png `
  --output runs/iris-segformer/visualizations/sample_comparison.png
```

#### 8.2 Loss and mIoU Curves
Generate the training-history plot from `metrics.csv`.

```powershell
python scripts/plot_training_curves.py `
  --metrics-file runs/iris-segformer/metrics.csv `
  --output runs/iris-segformer/visualizations/training_curves.png
```

If `metrics.csv` is missing, the script falls back to `epoch_summary.txt`.

#### 8.3 Generate Before/After Showcase Folders (for demo)
Generate the same folders used for reporting and sharing results:

```powershell
python scripts/generate_showcase_results.py --config configs/train.yaml --checkpoint runs/iris-segformer/checkpoints/best.pt --test-dir test --num-batch 10 --device cuda
```

This command fills:
- `inference_test_results/` (before/after masks, probability maps, boundaries)
- `overlay_test_results/` (overlay images + before/after comparison)
- `overlay_examples/` (multiple overlay styles + style grid + comprehensive comparison)
- `batch_test_results/` (batch predictions + `batch_summary.txt`)

If you don't have GPU, replace `--device cuda` with `--device cpu`.

### Step 9: Run FastAPI Upload UI (Optional)
After training, you can run an upload UI to send images and get predicted mask/overlay instantly.

```powershell
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

Open browser: `http://127.0.0.1:8000`

### Step 10: Clean Showcase Results Between Experiments
If you train many times and want to quickly reset all demo/result folders before a new run:

```powershell
python scripts/clean_showcase_results.py --yes
```

This clears content inside:
- `augmentation_visualizations/`
- `batch_test_results/`
- `inference_test_results/`
- `overlay_examples/`
- `overlay_test_results/`

To remove the folders themselves:

```powershell
python scripts/clean_showcase_results.py --yes --remove-dirs
```

---

## 📁 Project Structure

```
Eris/
├── requirements.txt              # Python dependencies
├── train.py                      # Training entrypoint script
├── inference.py                  # Run model inference on test images and save overlays
├── app_fastapi.py                # FastAPI upload UI for interactive inference
├── README.md                     # This documentation
├── .env                          # Kaggle credentials (create from .env.example)
├── .env.example                  # Template for .env
│
├── configs/
│   └── train.yaml               # Training hyperparameters & config
│
├── dataset/                      # Data directory (populated after steps 4-5)
│   ├── images/                   # Organized iris images (populated by step 5)
│   ├── masks/                    # Organized iris masks (populated by step 5)
│   ├── splits/                   # train.txt, val.txt, test.txt (created by step 5)
│   └── raw/                      # Raw extracted data (can delete after step 5)
│
├── scripts/
│   ├── download_ubiris.py        # Step 4: Download UBIRIS V2 from Kaggle
│   ├── download_dataset.py       # Generic dataset downloader (used by download_ubiris.py)
│   ├── prepare_dataset.py        # Step 5: Organize & split dataset
│   ├── visualize_prediction.py   # Step 8.1: Save image/GT/prediction comparison
│   ├── plot_training_curves.py   # Step 8.2: Save loss + mIoU plots
│   ├── visualize_augmentations.py # Create augmentation visualization artifacts
│   ├── evaluate_model.py         # Evaluate checkpoint and save metrics/failed cases
│   └── generate_showcase_results.py # Build before/after demo folders for sharing
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                     # Data loading & augmentation
│   │   ├── __init__.py
│   │   ├── ubiris_dataset.py    # UBIRIS V2 Dataset class (loads image+mask pairs)
│   │   ├── dataloader.py        # DataLoader factory (creates train/val loaders)
│   │   └── transforms.py        # Image augmentation pipeline (Albumentations+OpenCV)
│   │
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── heads.py             # Custom decoder heads (Boundary, Auxiliary, Attention)
│   │   └── segformer_custom.py  # SegFormer models (EnhancedSegFormer, DeepSupervisionSegFormer)
│   │
│   ├── losses/                   # Loss functions
│   │   ├── __init__.py
│   │   ├── dice.py              # Dice loss + tensor helpers
│   │   ├── focal.py             # Focal loss
│   │   ├── boundary.py          # Boundary-aware losses
│   │   ├── combined.py          # Combined loss presets
│   │   └── dice_focal_loss.py   # Backward-compatibility export layer
│   │
│   ├── evaluation/               # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py           # IoU/Dice/F1/boundary metrics + speed benchmark
│   │   └── evaluator.py         # Model evaluation orchestrator
│   │
│   ├── training/                 # Training loop
│   │   ├── __init__.py
│   │   └── trainer.py           # Trainer class (training, validation, checkpointing, logging)
│   │
│   ├── utils/                    # Helper utilities
│   │   ├── __init__.py
│   │   ├── config.py            # YAML config loading
│   │   ├── seed.py              # Random seed setting (reproducibility)
│   │   ├── metrics.py           # Dice, IoU metrics computation
│   │   ├── logger.py            # Logging setup
│   │   ├── checkpoint.py        # Checkpoint saving/loading
│   │   ├── visualization.py     # Prediction visualization helpers
│   │   ├── visualize.py         # Dataset preview + overlay export helpers
│   │   └── training_curves.py   # Metrics parsing + curve plotting helpers
│
├── augmentation_visualizations/   # Augmentation sample outputs for reporting
├── batch_test_results/            # Batch prediction outputs + summary
├── inference_test_results/        # Inference result artifacts (boundary/mask/prob)
├── overlay_examples/              # Overlay style examples
├── overlay_test_results/          # Overlay comparisons for before/after model
│
└── runs/                         # Output directory (auto-created in step 7)
    └── iris-segformer/           # Project output folder
        ├── checkpoints/          # Model checkpoints
        │   ├── best.pt          # Best validation model
        │   └── latest.pt        # Latest epoch model
        ├── logs/
        │   └── trainer.log      # Detailed training logs
    ├── epoch_summary.txt    # Per-epoch metrics (human-readable)
    ├── metrics.csv          # Structured metrics per epoch
    └── visualizations/
      ├── sample_comparison.png
      └── training_curves.png
```

---

## 🔧 Configuration Reference

### `configs/train.yaml` Detailed Options

```yaml
project:
  name: iris-segmentation                        # Project name (for logging)
  output_dir: runs/iris-segformer               # Where to save checkpoints, logs, metrics

data:
  root_dir: dataset                              # Dataset root directory
  image_dir: images                              # Image folder (relative to root_dir)
  mask_dir: masks                                # Mask folder (relative to root_dir)
  image_ext: .jpg                                # Image file extension
  mask_ext: .png                                 # Mask file extension
  train_split: 0.8                               # Train/val split ratio
  input_size: [512, 512]                         # Resize all images to this size
  num_workers: 4                                 # DataLoader workers (parallel loading)
  pin_memory: true                               # GPU memory pinning (faster loading)

model:
  backbone_name: nvidia/segformer-b0-finetuned-ade-512-512  # HuggingFace model
                                                 # Options: b0 (small), b1, b2, b3 (large)
  num_classes: 1                                 # Binary segmentation (iris vs background)
  decoder_channels: 256                          # Feature channels in decoder
  dropout: 0.1                                   # Dropout rate (regularization)

training:
  seed: 42                                       # Random seed for reproducibility
  device: cuda                                   # cuda (GPU) or cpu
  epochs: 2                                      # Number of training epochs
                                                 # Use 2-5 for quick testing
                                                 # Use 50+ for production training
  batch_size: 8                                  # Batch size (reduce if GPU runs out of memory)
  lr: 0.0001                                     # Learning rate (0.0001 is standard)
  weight_decay: 0.00001                          # L2 regularization (prevents overfitting)
  amp: true                                      # Automatic Mixed Precision (faster + less memory)
  grad_clip_norm: 1.0                            # Gradient clipping (prevents exploding gradients)
  save_every: 1                                  # Save checkpoint every N epochs

loss:
  dice_weight: 0.6                               # Dice loss contribution (60%)
  focal_weight: 0.4                              # Focal loss contribution (40%)
  focal_alpha: 0.8                               # Focal loss alpha parameter
  focal_gamma: 2.0                               # Focal loss gamma parameter

logging:
  log_interval: 10                               # Log progress every N batches
  metric_to_monitor: val_dice                   # Metric for best model selection
  monitor_mode: max                              # max (higher is better) or min
```

---

## 📊 Training Metrics Explained

The model tracks these metrics per epoch:

| Metric | Definition | Range | Interpretation |
|--------|-----------|-------|-----------------|
| `loss_train` | Training set loss (Dice + Focal) | 0 to ∞ | Lower is better; should decrease over epochs |
| `loss_val` | Validation set loss | 0 to ∞ | Lower is better; watch for overfitting |
| `acc_train` | Training Dice score | 0% to 100% | Higher is better; % overlap with ground truth |
| `acc_val` | Validation Dice score | 0% to 100% | Higher is better; measures generalization |
| `miou_train` | Training IoU score | 0% to 100% | Higher is better; more strict than Dice |
| `miou_val` | Validation IoU score | 0% to 100% | Higher is better; this is the curve to watch |
| `best` | Best validation Dice found so far | 0% to 100% | Best model saved when this improves |

**Example interpretation:**
```
Epoch 1: loss_train=4.3, acc_train=6%, loss_val=4.1, acc_val=9% (best: 9%)
Epoch 2: loss_train=3.1, acc_train=15%, loss_val=3.0, acc_val=18% (best: 18%)
```
✓ Both train and val metrics improving → Good training!
✗ Train improving but val stagnating → Overfitting (reduce epochs or increase augmentation)

---

## 🎨 Data Augmentation Pipeline

Training images go through aggressive augmentation to improve robustness:

| Stage | Augmentations | Purpose |
|-------|---------------|---------|
| **Geometric** | Rotation, shift, scale, flip | Handle natural iris variations |
| **Color** | Brightness, contrast, CLAHE | Normalize poor lighting conditions |
| **Blur** | Motion, Gaussian, Median | Simulate motion blur, focus blur |
| **Noise** | Gaussian, ISO, JPEG | Simulate sensor noise, compression |
| **Denoising** | Non-local means filter | Occasionally denoise to teach cleaning |
| **Normalization** | Mean/std normalization | Standardize pixel values for network |

**Validation set**: No augmentation, only resize + normalize (for fair evaluation)

---

## ⚠️ Troubleshooting

### Error: "CUDA out of memory"
**Solution:** Reduce batch size in `train.yaml`:
```yaml
training:
  batch_size: 4  # Was 8, reduced to 4
```

Or reduce input size:
```yaml
data:
  input_size: [384, 384]  # Was [512, 512], reduced
```

Or use CPU:
```yaml
training:
  device: cpu  # Use CPU instead
```

---

### Error: "No images found in dataset/images/"
**Solution:** Check that dataset preparation completed successfully:
```powershell
# Verify files exist
Get-ChildItem E:\MachineLearning\Eris\dataset\images | Measure-Object
Get-ChildItem E:\MachineLearning\Eris\dataset\masks | Measure-Object
```

If empty, re-run step 5:
```powershell
python scripts/prepare_dataset.py --source-images dataset/raw/images --source-masks dataset/raw/masks --copy
```

---

### Error: "Kaggle authentication failed"
**Solution:** Verify `.env` file is correct:
```powershell
# Check .env exists
Test-Path E:\MachineLearning\Eris\.env

# View content (redact key before sharing)
Get-Content E:\MachineLearning\Eris\.env
```

1. Get fresh API key from [https://www.kaggle.com/account/api](https://www.kaggle.com/account/api)
2. Update `.env` with new credentials
3. Test with: `python scripts/download_ubiris.py`

---

### Error: "Training is very slow"
**Solution:** Ensure you're using GPU:
```yaml
training:
  device: cuda  # Not cpu
```

Verify GPU is available:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

If GPU not available, check NVIDIA drivers and CUDA/cuDNN installation.

---

## 📈 Next Steps After Training

### Load Best Model for Inference
```python
import torch
from src.models import EnhancedSegFormer

# Initialize model
model = EnhancedSegFormer(
    backbone_name="nvidia/segformer-b0-finetuned-ade-512-512",
    num_classes=1
)

# Load best checkpoint
checkpoint = torch.load("runs/iris-segformer/checkpoints/best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict on new image
image = torch.randn(1, 3, 512, 512)  # Dummy image
with torch.no_grad():
    logits = model(image)
    mask = torch.sigmoid(logits) > 0.5  # Binary threshold
```

### Further Improvements
1. **Train longer**: Change `epochs: 2` → `epochs: 100` for better accuracy
2. **Use bigger backbone**: `segformer-b1` or `b2` instead of `b0`
3. **Fine-tune**: Use pretrained model as starting point for your own iris dataset
4. **Ensemble**: Train multiple models and ensemble predictions
5. **Post-processing**: Apply morphological operations to smooth predictions

---

## 📚 References

- [SegFormer Paper](https://arxiv.org/abs/2105.15203) - Original SegFormer architecture
- [UBIRIS V2 Dataset](https://iris.di.ubi.pt/) - Iris image dataset
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Model hub
- [Albumentations](https://albumentations.ai/) - Image augmentation library
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Deep learning framework
