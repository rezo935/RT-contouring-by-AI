# Installation Guide

This guide provides detailed instructions for installing the RT Auto-Contouring system on Windows.

## System Requirements

### Hardware
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 32 GB minimum (64 GB recommended for training)
- **GPU**: NVIDIA GPU with 11+ GB VRAM (RTX 3080/A5000 or better recommended)
- **Storage**: 500 GB+ free space for datasets and models

### Software
- **Operating System**: Windows 10/11 (64-bit)
- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or higher (for GPU support)
- **Git**: For version control

## Step 1: Install Python

1. Download Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install for all users"
   - Choose "Customize installation"
   - ✅ Select "pip" and "py launcher"
3. Verify installation:
   ```bash
   python --version
   pip --version
   ```

## Step 2: Install CUDA and cuDNN (for GPU)

### Install NVIDIA CUDA Toolkit

1. Download CUDA Toolkit 11.8 from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the wizard
3. Verify installation:
   ```bash
   nvcc --version
   ```

### Install cuDNN

1. Download cuDNN from [NVIDIA website](https://developer.nvidia.com/cudnn) (requires free account)
2. Extract the archive
3. Copy files to CUDA installation directory:
   - `bin` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - `include` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   - `lib` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib`

## Step 3: Clone the Repository

```bash
git clone https://github.com/rezo935/RT-contouring-by-AI.git
cd RT-contouring-by-AI
```

## Step 4: Create Virtual Environment

Using venv (built-in):
```bash
python -m venv venv
venv\Scripts\activate
```

Or using conda:
```bash
conda create -n rt-contouring python=3.10
conda activate rt-contouring
```

## Step 5: Install PyTorch with CUDA Support

Install PyTorch with CUDA 11.8 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is available:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Step 6: Install Project Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

## Step 7: Verify Installation

Run a quick verification:

```python
python -c "
import torch
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder
import nnunetv2

print('✅ All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'nnU-Net version: {nnunetv2.__version__}')
"
```

## Step 8: Configure Environment

### Set up nnU-Net directories

Create a configuration file or set environment variables:

**Option A: Environment variables (recommended for permanent setup)**

Add to Windows Environment Variables:
- `nnUNet_raw`: `D:\RT-Contouring\data\nnUNet\nnUNet_raw`
- `nnUNet_preprocessed`: `D:\RT-Contouring\data\nnUNet\nnUNet_preprocessed`
- `nnUNet_results`: `D:\RT-Contouring\data\nnUNet\nnUNet_results`

**Option B: Set in Python**

The project handles this automatically through `config/paths.py`, but you can override:

```python
from config.paths import get_path_config

path_config = get_path_config(
    data_root=r"D:\RT-Contouring\data",
    nnunet_root=r"D:\RT-Contouring\data\nnUNet"
)

path_config.setup_nnunet_env()
path_config.create_directories()
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: 
- Reduce batch size in nnU-Net configuration
- Use 3d_lowres configuration instead of 3d_fullres
- Upgrade GPU or use a machine with more VRAM

### Issue: Import errors for rt-utils

**Solution**:
```bash
pip install --upgrade rt-utils
```

### Issue: nnU-Net cannot find datasets

**Solution**:
- Verify environment variables are set correctly
- Check that dataset folders follow nnU-Net naming convention: `Dataset001_Name`
- Ensure `dataset.json` exists in the dataset folder

### Issue: Slow preprocessing

**Solution**:
- Increase number of processes: `--num-processes 16`
- Use SSD storage for better I/O performance
- Ensure sufficient RAM is available

### Issue: Permission errors on Windows

**Solution**:
- Run terminal as Administrator
- Check antivirus isn't blocking file operations
- Ensure you have write permissions to data directories

## Optional: Install Development Tools

For development and testing:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Optional: Install Jupyter Notebook Support

For interactive data exploration:

```bash
pip install -e ".[notebooks]"
```

Then launch Jupyter:

```bash
jupyter notebook
```

## Next Steps

After successful installation, you're ready to start training!

**👉 Follow the [GETTING STARTED GUIDE](GETTING_STARTED.md)** for a complete step-by-step workflow from installation to training.

Or review individual documentation:

1. [DATA_PREPARATION.md](DATA_PREPARATION.md) - Organize and prepare your dataset
2. [TRAINING.md](TRAINING.md) - Train your models
3. Explore the example notebook: `notebooks/01_data_exploration.ipynb`

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review nnU-Net documentation: https://github.com/MIC-DKFZ/nnUNet
3. Check project issues on GitHub
4. Consult the rt-utils documentation: https://github.com/qurit/rt-utils

## Version Information

Tested with:
- Python 3.10.11
- PyTorch 2.0.1 + CUDA 11.8
- nnU-Net v2.2
- Windows 11

Last updated: February 2026
