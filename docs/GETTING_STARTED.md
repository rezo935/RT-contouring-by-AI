# Getting Started - Next Steps After Installation

Congratulations on installing the RT Auto-Contouring system! This guide will help you go from installation to training your first model.

## Prerequisites Checklist

Before starting, verify you have completed:

- ✅ **Installation Complete**: Followed [INSTALLATION.md](INSTALLATION.md)
- ✅ **GPU Available**: NVIDIA GPU with CUDA support
- ✅ **DICOM Data Ready**: CT scans with RTSTRUCT contours
- ✅ **Virtual Environment Active**: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux)

### Quick Verification

Run these commands to verify your setup:

```bash
# Check Python environment
python --version  # Should be 3.10 or 3.11

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify all packages installed
python -c "import nnunetv2, pydicom, SimpleITK; print('✅ All packages OK')"
```

## Overview: Training Workflow

The complete workflow consists of 4 main steps:

```
1. Organize DICOM Data    →    2. Quality Assessment    →    3. Preprocessing    →    4. Training
   (5 minutes)                    (10-30 minutes)              (20-60 minutes)         (24-48 hours)
```

## Step-by-Step Guide

### Step 1: Organize Your DICOM Data

Create the following directory structure:

```
data/
└── dicom_raw/
    ├── Patient_001/
    │   ├── CT/
    │   │   ├── CT.1.dcm
    │   │   ├── CT.2.dcm
    │   │   └── ...
    │   └── RS.1.dcm  (RTSTRUCT file)
    ├── Patient_002/
    │   ├── CT/
    │   └── RS.1.dcm
    └── ...
```

**Required structures in your RTSTRUCT:**
- Bladder
- Rectum
- Femoral Head Left
- Femoral Head Right
- Optional: Bowel, Penile Bulb, Vaginal Canal

📖 **Detailed info**: See [DATA_PREPARATION.md](DATA_PREPARATION.md#data-organization)

---

### Step 2: Assess Data Quality

Evaluate your dataset to identify potential issues:

```bash
python scripts/run_quality_assessment.py \
    data/dicom_raw \
    --output-dir data/quality_reports \
    --visualize \
    --select-cohort \
    --min-quality-score 70.0
```

**What this does:**
- Checks for missing structures
- Identifies volume outliers
- Generates quality report and training cohort selection
- Creates visualization plots

**Output:** 
- `data/quality_reports/quality_assessment.csv` - Detailed metrics
- `data/quality_reports/training_cohort.txt` - Selected cases for training
- `data/quality_reports/*.png` - Visualization plots

**Review the results** and ensure you have at least 20-30 good quality cases.

📖 **Detailed info**: See [DATA_PREPARATION.md](DATA_PREPARATION.md#step-1-quality-assessment)

---

### Step 3: Preprocess Data

Convert DICOM to nnU-Net format:

```bash
python scripts/run_preprocessing.py full \
    data/dicom_raw \
    --dataset-id 1 \
    --dataset-name PelvisOAR \
    --cohort-file data/quality_reports/training_cohort.txt \
    --train-val-split 0.8
```

**What this does:**
- Converts DICOM CT and RTSTRUCT to NIfTI format
- Creates nnU-Net dataset structure
- Splits data into training and validation sets

**Output:**
- `data/nnUNet/nnUNet_raw/Dataset001_PelvisOAR/` - nnU-Net formatted dataset

Now run nnU-Net preprocessing:

```bash
python scripts/run_training.py preprocess \
    --dataset-id 1 \
    --verify \
    --num-processes 8
```

**What this does:**
- Analyzes dataset properties (intensities, spacings, sizes)
- Determines optimal configurations (2D, 3D lowres, 3D fullres)
- Creates preprocessed data cache

**Duration:** 10-30 minutes depending on dataset size

📖 **Detailed info**: See [DATA_PREPARATION.md](DATA_PREPARATION.md#step-2-data-conversion) and [TRAINING.md](TRAINING.md#step-1-nnu-net-preprocessing)

---

### Step 4: Start Training

#### Option A: Quick Test (Single Fold)

Train just one fold to test everything works:

```bash
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --fold 0
```

**Duration:** 24-48 hours (depending on GPU and dataset size)

#### Option B: Full Training (Recommended)

Train all 5 folds for best performance:

```bash
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --all-folds
```

**Duration:** 5-10 days (5× single fold time)

**💡 Tip:** Start with Option A first to ensure everything works, then run Option B.

📖 **Detailed info**: See [TRAINING.md](TRAINING.md#step-2-training)

---

### Step 5: Monitor Training

Check training status:

```bash
python scripts/run_training.py status \
    --dataset-id 1 \
    --configuration 3d_fullres
```

Training logs are saved in:
```
data/nnUNet/nnUNet_results/Dataset001_PelvisOAR.../fold_0/training_log.txt
```

---

### Step 6: Run Inference (After Training)

Once training completes, test your model on a new case:

```bash
python scripts/run_inference.py \
    path/to/new/ct/dicom \
    --dataset-id 1 \
    --create-rtstruct \
    --output-dir results/
```

This will generate auto-contoured RTSTRUCT that you can import into your TPS.

---

## Quick Reference: Complete Workflow

Here's the full sequence of commands:

```bash
# 1. Quality Assessment
python scripts/run_quality_assessment.py data/dicom_raw --output-dir data/quality_reports --visualize --select-cohort --min-quality-score 70.0

# 2. Data Preprocessing
python scripts/run_preprocessing.py full data/dicom_raw --dataset-id 1 --dataset-name PelvisOAR --cohort-file data/quality_reports/training_cohort.txt --train-val-split 0.8

# 3. nnU-Net Preprocessing
python scripts/run_training.py preprocess --dataset-id 1 --verify --num-processes 8

# 4. Training (Quick Test)
python scripts/run_training.py train --dataset-id 1 --configuration 3d_fullres --fold 0

# OR Training (Full - Recommended)
python scripts/run_training.py train --dataset-id 1 --configuration 3d_fullres --all-folds

# 5. Check Status
python scripts/run_training.py status --dataset-id 1 --configuration 3d_fullres

# 6. Run Inference
python scripts/run_inference.py path/to/ct/dicom --dataset-id 1 --create-rtstruct --output-dir results/
```

---

## Common Issues and Solutions

### Issue: "CUDA out of memory"

**Solution:** Use a lower resolution configuration:
```bash
python scripts/run_training.py train --dataset-id 1 --configuration 3d_lowres --fold 0
```

### Issue: "Dataset not found"

**Solution:** Verify dataset was created correctly:
```bash
ls data/nnUNet/nnUNet_raw/Dataset001_PelvisOAR/
# Should show: imagesTr/, labelsTr/, dataset.json
```

### Issue: "Structure names not recognized"

**Solution:** Check supported names in `config/structures.py` or add custom aliases.

### Issue: Training very slow

**Solutions:**
- Reduce `--num-processes` during preprocessing
- Use `3d_lowres` configuration
- Verify GPU is being used: `nvidia-smi`

---

## Minimum Dataset Requirements

- **Minimum**: 20-30 cases for proof-of-concept
- **Recommended**: 50-100 cases for good performance  
- **Optimal**: 100+ cases for production use

Expected Dice scores with 50-100 cases:
- Bladder: 0.93-0.96
- Rectum: 0.91-0.94
- Femoral Heads: 0.94-0.97
- Bowel: 0.78-0.85

---

## Hardware Requirements

### For Training:
- **GPU**: NVIDIA GPU with 11+ GB VRAM (RTX 3080/3090, A5000 or better)
  - 3d_fullres: 11+ GB VRAM
  - 3d_lowres: 8+ GB VRAM
  - 2d: 6+ GB VRAM
- **RAM**: 32 GB minimum (64 GB recommended)
- **Storage**: 500+ GB for datasets and models

### For Inference Only:
- **GPU**: NVIDIA GPU with 8+ GB VRAM
- **RAM**: 16 GB minimum

---

## Next Steps

Once you have trained your model:

1. **Validate Performance**: Review cross-validation results in `data/nnUNet/nnUNet_results/`
2. **Test on New Cases**: Run inference and visually inspect results
3. **Deploy API Service**: `python -m src.service.pipeline` for integration with PACS/TPS
4. **Continuous Improvement**: Add more training cases and retrain periodically

---

## Additional Resources

- 📘 [INSTALLATION.md](INSTALLATION.md) - Detailed installation guide
- 📗 [DATA_PREPARATION.md](DATA_PREPARATION.md) - Comprehensive data preparation guide
- 📙 [TRAINING.md](TRAINING.md) - Complete training guide with advanced topics
- 📊 [notebooks/01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb) - Interactive data exploration

---

## Getting Help

If you encounter issues:

1. Check the troubleshooting sections in each documentation file
2. Review nnU-Net documentation: https://github.com/MIC-DKFZ/nnUNet
3. Open an issue on GitHub with:
   - Your system specs (GPU, RAM, OS)
   - Error messages and logs
   - Steps to reproduce the issue

---

**Ready to start?** Jump to [Step 1: Organize Your DICOM Data](#step-1-organize-your-dicom-data)!

---

*Last Updated: February 2026*
