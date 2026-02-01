# Data Preparation Guide

This guide explains how to prepare your radiotherapy DICOM data for training with nnU-Net.

## Overview

The data preparation pipeline consists of three main steps:

1. **Quality Assessment**: Evaluate dataset quality and select training cohort
2. **DICOM to NIfTI Conversion**: Convert DICOM CT and RTSTRUCT to NIfTI format
3. **nnU-Net Dataset Preparation**: Create properly formatted nnU-Net dataset

## Prerequisites

- DICOM CT series with corresponding RTSTRUCT files
- Structures contoured in RTSTRUCT files
- Completed [installation](INSTALLATION.md)

## Data Organization

### Input DICOM Structure

Organize your DICOM data as follows:

```
data/dicom_raw/
├── Patient_001/
│   ├── CT/
│   │   ├── CT.1.dcm
│   │   ├── CT.2.dcm
│   │   └── ...
│   └── RS.1.dcm  (RTSTRUCT file)
├── Patient_002/
│   ├── CT/
│   │   └── ...
│   └── RS.1.dcm
└── ...
```

**Notes:**
- Each patient should have their own directory
- CT slices can be in a subdirectory or directly in patient directory
- RTSTRUCT file should be in the patient directory
- File names can vary; the scripts auto-detect CT series and RTSTRUCT

### Required Structures

For pelvis OAR segmentation, the following structures should be contoured:

**Essential (for training)**:
- Bladder
- Rectum
- Femoral Head Left
- Femoral Head Right

**Optional but recommended**:
- Bowel (Bowel Bag)
- Penile Bulb (male patients)
- Vaginal Canal (female patients)

**Future**:
- Lymph Nodes

### Structure Naming

The system handles various naming conventions automatically. For example, "Bladder" can be:
- `bladder`, `Bladder`, `BLADDER`
- `bladder_o`, `Bladder_O` (with optimization suffix)
- `vesica`, `Vesica` (Latin name)
- `urinary_bladder`, etc.

See `config/structures.py` for complete list of supported aliases.

## Step 1: Quality Assessment

Assess your dataset quality before training:

```bash
python scripts/run_quality_assessment.py \
    data/dicom_raw \
    --output-dir data/quality_reports \
    --visualize \
    --select-cohort \
    --min-quality-score 70.0 \
    --required-structures bladder rectum
```

### Parameters

- `data/dicom_raw`: Path to DICOM root directory
- `--output-dir`: Where to save quality reports
- `--visualize`: Generate visualization plots
- `--select-cohort`: Select training cohort based on criteria
- `--min-quality-score`: Minimum quality score threshold (0-100)
- `--required-structures`: Structures that must be present

### Outputs

Quality assessment creates:
- `quality_assessment.csv`: Detailed metrics for all cases
- `training_cohort.txt`: List of selected case IDs
- `volume_distributions.png`: Volume distributions by structure
- `quality_scores.png`: Quality score histogram
- `structure_presence.png`: Heatmap of structure presence

### Quality Criteria

Cases are flagged for issues such as:
- **Missing structures**: Required structures not found
- **Outlier volumes**: Volumes outside expected ranges
- **Disconnected components**: Structures split into multiple parts
- **Empty contours**: Zero-volume structures

### Reviewing Results

1. Check `quality_assessment.csv` for detailed metrics
2. Review visualizations to identify outliers
3. Manually inspect flagged cases
4. Update `training_cohort.txt` if needed

## Step 2: Data Conversion

Convert DICOM to NIfTI format suitable for nnU-Net.

### Option A: Full Pipeline (Recommended)

Convert and prepare dataset in one command:

```bash
python scripts/run_preprocessing.py full \
    data/dicom_raw \
    --dataset-id 1 \
    --dataset-name PelvisOAR \
    --cohort-file data/quality_reports/training_cohort.txt \
    --train-val-split 0.8
```

### Option B: Step-by-Step

**Step 2a: Convert DICOM to NIfTI**

```bash
python scripts/run_preprocessing.py convert \
    data/dicom_raw \
    --output-dir data/dicom_processed/nifti \
    --cohort-file data/quality_reports/training_cohort.txt
```

**Step 2b: Prepare nnU-Net Dataset**

```bash
python scripts/run_preprocessing.py prepare \
    data/dicom_processed/nifti \
    --dataset-id 1 \
    --dataset-name PelvisOAR \
    --description "Pelvis OAR Segmentation" \
    --modality CT \
    --train-val-split 0.8
```

### Parameters

- `--dataset-id`: Unique integer ID (e.g., 1 for Dataset001)
- `--dataset-name`: Dataset name (e.g., PelvisOAR)
- `--cohort-file`: Optional file with case IDs to process
- `--train-val-split`: Fraction for training (rest for validation)

### Dataset Naming Convention

nnU-Net requires specific naming:
- Dataset folder: `Dataset{ID:03d}_{Name}` (e.g., `Dataset001_PelvisOAR`)
- Image files: `{CaseID}_0000.nii.gz` (0000 = first channel)
- Label files: `{CaseID}.nii.gz`

## Step 3: Verify Dataset

After preparation, verify the dataset:

```python
from config.paths import get_path_config
from src.preprocessing.prepare_dataset import NnUNetDatasetPreparator

path_config = get_path_config()
preparator = NnUNetDatasetPreparator(path_config)

# Verify dataset
is_valid = preparator.verify_dataset(dataset_id=1, dataset_name="PelvisOAR")

# Get statistics
stats = preparator.get_dataset_statistics(dataset_id=1, dataset_name="PelvisOAR")
print(stats)
```

### Expected Structure

```
data/nnUNet/nnUNet_raw/Dataset001_PelvisOAR/
├── dataset.json
├── imagesTr/
│   ├── Patient_001_0000.nii.gz
│   ├── Patient_002_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── Patient_001.nii.gz
│   ├── Patient_002.nii.gz
│   └── ...
└── splits_final.json
```

### dataset.json Contents

```json
{
  "name": "PelvisOAR",
  "description": "Pelvis OAR Segmentation",
  "modality": {
    "0": "CT"
  },
  "labels": {
    "background": 0,
    "bladder": 1,
    "rectum": 2,
    "bowel": 3,
    "femoral_head_left": 4,
    "femoral_head_right": 5,
    "penile_bulb": 6,
    "vaginal_canal": 7,
    "lymph_nodes": 8
  },
  "numTraining": 50,
  "file_ending": ".nii.gz"
}
```

## Data Augmentation

nnU-Net applies data augmentation automatically during training:
- Random rotations
- Random scaling
- Random elastic deformations
- Brightness and contrast adjustments
- Gaussian noise
- Gamma correction

No manual augmentation is needed.

## Best Practices

### Dataset Size
- **Minimum**: 20-30 cases for proof-of-concept
- **Recommended**: 50-100 cases for good performance
- **Optimal**: 100+ cases for production use

### Data Quality
- Ensure consistent CT protocols
- Use complete structure sets
- Verify contour quality manually
- Remove cases with:
  - Incomplete coverage
  - Severe artifacts
  - Incorrect contours
  - Technical failures

### Structure Consistency
- Use consistent structure names across cases
- Ensure all required structures are present
- Check for anatomical variations
- Consider separate models for male/female anatomy if needed

### Train/Validation Split
- Default 80/20 split usually works well
- Ensure validation set is representative
- Consider stratification by:
  - Institution/scanner
  - Patient demographics
  - Disease characteristics

## Troubleshooting

### Issue: Structure names not recognized

**Solution**: Add aliases to `config/structures.py`:
```python
STRUCTURE_ALIASES = {
    "bladder": [
        "bladder", "Bladder", "BLADDER",
        "your_custom_name"  # Add here
    ]
}
```

### Issue: Missing CT or RTSTRUCT files

**Solution**: 
- Check directory structure matches expected format
- Verify DICOM files are not corrupted
- Ensure RTSTRUCT references the correct CT series

### Issue: Empty or tiny structures

**Solution**:
- Review contours in treatment planning system
- Check for contouring errors
- Consider excluding cases with poor quality contours

### Issue: Volume outliers

**Solution**:
- Manually review flagged cases
- Update expected volume ranges in `config/structures.py` if needed
- Consider excluding extreme outliers

## Next Steps

After data preparation:
1. Review [TRAINING.md](TRAINING.md) for training workflows
2. Start with nnU-Net preprocessing: `python scripts/run_training.py preprocess --dataset-id 1`
3. Train your first model: `python scripts/run_training.py train --dataset-id 1 --fold 0`

## Additional Resources

- nnU-Net documentation: https://github.com/MIC-DKFZ/nnUNet
- DICOM standard: https://www.dicomstandard.org/
- RT-STRUCT specification: https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.19.html
