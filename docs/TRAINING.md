# Training Guide

This guide explains how to train nnU-Net models for radiotherapy auto-contouring.

## Overview

The training workflow consists of:

1. **Preprocessing**: nnU-Net analyzes dataset and creates preprocessed data
2. **Training**: Train models using 5-fold cross-validation
3. **Validation**: Evaluate model performance
4. **Selection**: Choose best configuration for deployment

## Prerequisites

- Completed [data preparation](DATA_PREPARATION.md)
- nnU-Net dataset properly formatted
- GPU with sufficient VRAM (11+ GB recommended)

## Training Workflow

### Step 1: nnU-Net Preprocessing

Before training, nnU-Net must analyze your dataset:

```bash
python scripts/run_training.py preprocess \
    --dataset-id 1 \
    --verify \
    --num-processes 8
```

**Parameters:**
- `--dataset-id`: Dataset ID (e.g., 1 for Dataset001)
- `--verify`: Verify dataset integrity before preprocessing
- `--num-processes`: Number of parallel processes (adjust based on CPU cores)

**What happens:**
- Dataset fingerprint extraction (image properties, intensity statistics)
- Automatic configuration planning (2D, 3D lowres, 3D fullres)
- Preprocessing and resampling of data
- Creation of preprocessed cache

**Duration:** 10-30 minutes depending on dataset size

**Output:** Preprocessed data in `data/nnUNet/nnUNet_preprocessed/Dataset001_PelvisOAR/`

### Step 2: Training

#### Option A: Train Single Fold (For Testing)

Train fold 0 only (fastest, for testing):

```bash
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --fold 0
```

**Duration:** 24-48 hours depending on GPU and dataset size

#### Option B: Train All Folds (Recommended)

Train all 5 folds for cross-validation:

```bash
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --all-folds
```

**Duration:** 5-10 days (5× single fold duration)

**Recommended:** Run overnight or over a weekend

#### Option C: Train Multiple Configurations

Train different configurations to find the best:

```bash
# 3D full resolution
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --all-folds

# 3D low resolution  
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_lowres \
    --all-folds

# 2D
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 2d \
    --all-folds
```

### Training Parameters

**Dataset and Configuration:**
- `--dataset-id`: Dataset ID
- `--configuration`: Model configuration
  - `3d_fullres`: 3D full resolution (most accurate, highest VRAM)
  - `3d_lowres`: 3D low resolution (faster, less VRAM)
  - `2d`: 2D (fastest, lowest VRAM)
- `--fold`: Fold number (0-4) or use `--all-folds`

**Advanced Options:**
- `--trainer`: Trainer class (default: `nnUNetTrainer`)
- `--plans`: Plans identifier (default: `nnUNetPlans`)
- `--num-epochs`: Override default number of epochs
- `--continue-training`: Resume from checkpoint

### Monitoring Training

#### Check Training Status

```bash
python scripts/run_training.py status \
    --dataset-id 1 \
    --configuration 3d_fullres
```

#### TensorBoard (if enabled)

nnU-Net can log to TensorBoard:

```bash
tensorboard --logdir data/nnUNet/nnUNet_results/Dataset001_PelvisOAR
```

#### Training Logs

Logs are saved in:
```
data/nnUNet/nnUNet_results/Dataset001_PelvisOAR_nnUNetTrainer__nnUNetPlans__3d_fullres/
├── fold_0/
│   ├── training_log.txt
│   ├── checkpoint_best.pth
│   ├── checkpoint_final.pth
│   └── validation_raw/
├── fold_1/
└── ...
```

### Step 3: Find Best Configuration

After training multiple configurations:

```bash
python scripts/run_training.py find-best \
    --dataset-id 1 \
    --configurations 3d_fullres 3d_lowres 2d
```

This compares validation performance across configurations and recommends the best.

## Understanding nnU-Net Configurations

### 3D Full Resolution (`3d_fullres`)

**Characteristics:**
- Operates on full 3D volumes
- Highest accuracy
- Best for capturing 3D context

**Requirements:**
- GPU: 11+ GB VRAM
- Training time: Longest
- Inference time: ~1-2 minutes per case

**Best for:** Final production models

### 3D Low Resolution (`3d_lowres`)

**Characteristics:**
- Operates on downsampled 3D volumes
- Good balance of speed and accuracy
- Useful for initial experiments

**Requirements:**
- GPU: 8+ GB VRAM
- Training time: Moderate
- Inference time: ~30-60 seconds per case

**Best for:** Quick iterations, lower-end GPUs

### 2D (`2d`)

**Characteristics:**
- Processes individual 2D slices
- Fastest training and inference
- May miss some 3D context

**Requirements:**
- GPU: 6+ GB VRAM
- Training time: Shortest
- Inference time: ~20-30 seconds per case

**Best for:** Rapid prototyping, limited hardware

### Cascade (Advanced)

For very large structures, nnU-Net may automatically create a cascade:
1. `3d_lowres`: Coarse segmentation
2. `3d_cascade_fullres`: Refinement at full resolution

## Training Tips

### GPU Memory Management

**If you get CUDA Out of Memory errors:**

1. Use lower resolution configuration:
   ```bash
   --configuration 3d_lowres
   ```

2. Reduce batch size (requires modifying nnU-Net plans)

3. Use gradient checkpointing (advanced)

4. Upgrade GPU or use cloud resources

### Resuming Training

If training is interrupted:

```bash
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --fold 0 \
    --continue-training
```

### Training on Multiple GPUs

nnU-Net supports multi-GPU training (requires code modification). See nnU-Net documentation.

### Hyperparameter Tuning

nnU-Net automatically determines most hyperparameters. Advanced users can modify:
- `nnUNet_preprocessed/Dataset001_PelvisOAR/nnUNetPlans.json`

Common modifications:
- Patch size
- Batch size
- Learning rate schedule

## Validation and Evaluation

### Cross-Validation Results

After training all folds, results are summarized in:
```
data/nnUNet/nnUNet_results/Dataset001_PelvisOAR_.../crossval_results_folds_0_1_2_3_4/
```

Metrics include:
- **Dice Score**: Overlap between prediction and ground truth (0-1, higher is better)
- **Hausdorff Distance 95**: Surface distance metric (mm, lower is better)
- **Sensitivity**: True positive rate
- **Precision**: Positive predictive value

### Per-Structure Metrics

Results are provided for each structure:

| Structure | Dice | HD95 (mm) | Sensitivity | Precision |
|-----------|------|-----------|-------------|-----------|
| Bladder | 0.95 | 2.5 | 0.96 | 0.94 |
| Rectum | 0.93 | 3.2 | 0.94 | 0.92 |
| ... | ... | ... | ... | ... |

### Interpreting Results

**Dice Score Guidelines:**
- > 0.90: Excellent
- 0.80-0.90: Good
- 0.70-0.80: Acceptable
- < 0.70: Needs improvement

**Structure-specific expectations:**
- Bladder, Rectum: Expect Dice > 0.90
- Femoral Heads: Expect Dice > 0.92
- Bowel: Expect Dice > 0.75 (more variable)
- Penile Bulb: Expect Dice > 0.70 (small structure)

## Common Issues

### Issue: Training loss not decreasing

**Possible causes:**
- Learning rate too high/low
- Insufficient data
- Poor data quality

**Solutions:**
- Check data quality
- Review quality assessment results
- Consider data augmentation settings

### Issue: Overfitting

**Symptoms:**
- Training loss decreases but validation loss increases
- Large gap between training and validation performance

**Solutions:**
- Increase dataset size
- Enable stronger augmentation (modify plans)
- Early stopping (nnU-Net does this automatically)

### Issue: Poor performance on specific structure

**Solutions:**
1. Check structure is consistently present in dataset
2. Review contour quality for that structure
3. Adjust structure label mapping
4. Consider training separate model

### Issue: Out of memory during validation

**Solution:**
- Reduce number of validation cases processed at once
- Use `--validation_only` flag with smaller batch

## Advanced Topics

### Custom Trainers

Create custom trainer by extending `nnUNetTrainer`:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class MyCustomTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, ...):
        super().__init__(plans, configuration, fold, ...)
        # Custom initialization
```

### Ensemble Inference

By default, nnU-Net uses all 5 folds for ensemble inference (best accuracy).

For faster inference, use single fold:
```bash
--folds 0  # Use only fold 0
```

### Transfer Learning

Fine-tune pre-trained model on new data:
1. Train base model on large dataset
2. Copy weights to new dataset
3. Train with lower learning rate and fewer epochs

## Performance Benchmarks

**Typical performance for pelvis OAR (50-100 training cases):**

| Structure | Expected Dice | Training Time (3d_fullres) |
|-----------|---------------|---------------------------|
| Bladder | 0.93-0.96 | ~36 hours (all folds) |
| Rectum | 0.91-0.94 | ~36 hours (all folds) |
| Femoral Heads | 0.94-0.97 | ~36 hours (all folds) |
| Bowel | 0.78-0.85 | ~36 hours (all folds) |

*Times are for RTX 3090 with 50 cases*

## Next Steps

After training:

1. Validate model performance on held-out test set
2. Run inference on new cases: `python scripts/run_inference.py`
3. Deploy via API service: `python -m src.service.pipeline`
4. Integrate with PACS/TPS workflow

## Resources

- nnU-Net paper: https://arxiv.org/abs/1904.08128
- nnU-Net repository: https://github.com/MIC-DKFZ/nnUNet
- nnU-Net documentation: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/
