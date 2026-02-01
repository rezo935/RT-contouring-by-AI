# RT Auto-Contouring by AI

Automated radiotherapy organ-at-risk (OAR) contouring using deep learning with nnU-Net for pelvis structures.

## Overview

This project provides a complete pipeline for automated segmentation of pelvic organs at risk (OARs) in radiotherapy planning, using the state-of-the-art nnU-Net deep learning framework.

### Target Structures (Pelvis)

- **Bladder** ✓
- **Rectum** ✓
- **Bowel (Bowel Bag)** ✓
- **Femoral Head Left** ✓
- **Femoral Head Right** ✓
- **Penile Bulb** (male) ✓
- **Vaginal Canal** (female) ✓
- **Lymph Nodes** (future)

### Key Features

- 🔍 **Quality Assessment**: Automated dataset quality evaluation with volume checks and outlier detection
- 🔄 **DICOM Integration**: Full support for DICOM CT and RTSTRUCT import/export
- 🧠 **nnU-Net v2**: State-of-the-art deep learning segmentation
- 📊 **Comprehensive Workflow**: From raw DICOM to production-ready contours
- 🚀 **FastAPI Service**: REST API for integration with PACS/TPS systems
- 📈 **Quality Reports**: Detailed metrics and visualizations
- 🎨 **Eclipse-Compatible**: Structure colors matching TPS conventions

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rezo935/RT-contouring-by-AI.git
cd RT-contouring-by-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

For detailed installation instructions, see [INSTALLATION.md](docs/INSTALLATION.md).

### ✨ Next Steps After Installation

**New to the project?** Follow our step-by-step guide:

👉 **[GETTING STARTED GUIDE](docs/GETTING_STARTED.md)** - Complete workflow from installation to training

This guide walks you through:
- Organizing your DICOM data
- Running quality assessment
- Preprocessing datasets
- Training your first model
- Running inference

### Basic Usage

#### 1. Quality Assessment

```bash
python scripts/run_quality_assessment.py \
    data/dicom_raw \
    --visualize \
    --select-cohort \
    --min-quality-score 70.0
```

#### 2. Data Preprocessing

```bash
python scripts/run_preprocessing.py full \
    data/dicom_raw \
    --dataset-id 1 \
    --dataset-name PelvisOAR \
    --cohort-file data/quality_reports/training_cohort.txt
```

#### 3. Training

```bash
# Preprocess
python scripts/run_training.py preprocess --dataset-id 1

# Train
python scripts/run_training.py train \
    --dataset-id 1 \
    --configuration 3d_fullres \
    --all-folds
```

#### 4. Inference

```bash
python scripts/run_inference.py \
    path/to/ct/dicom \
    --dataset-id 1 \
    --create-rtstruct \
    --output-dir results/
```

## Documentation

- 🚀 **[Getting Started Guide](docs/GETTING_STARTED.md)** - Quick path from installation to training (START HERE!)
- 📘 [Installation Guide](docs/INSTALLATION.md) - Detailed setup instructions for Windows
- 📗 [Data Preparation](docs/DATA_PREPARATION.md) - How to prepare and organize your training data
- 📙 [Training Guide](docs/TRAINING.md) - Complete training workflow and best practices

## Project Structure

```
RT-contouring-by-AI/
├── config/                      # Configuration modules
│   ├── structures.py           # Structure definitions, aliases, colors
│   └── paths.py                # Path configuration
├── src/                        # Source code
│   ├── data_quality/          # Dataset quality assessment
│   ├── dicom_listener/        # DICOM Storage SCP for receiving files
│   ├── preprocessing/         # DICOM to NIfTI conversion
│   ├── training/              # nnU-Net training wrapper
│   ├── inference/             # Inference and prediction
│   └── service/               # FastAPI service
├── scripts/                    # CLI scripts
│   ├── run_quality_assessment.py
│   ├── run_preprocessing.py
│   ├── run_training.py
│   ├── run_inference.py
│   └── run_dicom_listener.py
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── docs/                       # Documentation
│   ├── INSTALLATION.md
│   ├── DATA_PREPARATION.md
│   └── TRAINING.md
├── requirements.txt            # Python dependencies
└── setup.py                   # Package setup
```

## DICOM Listener

The DICOM Storage SCP (Service Class Provider) allows you to receive CT and RTSTRUCT files directly from Eclipse TPS or other DICOM sources for training data collection.

### Starting the Listener

```bash
python scripts/run_dicom_listener.py --ae-title AUTOCONTOUR --port 11112
```

Command-line options:
- `--ae-title`: AE Title for the DICOM SCP (default: AUTOCONTOUR)
- `--port`: Port number to listen on (default: 11112)
- `--output`: Output directory for received files (default: platform-dependent - see below)
- `--ip`: Local IP address to display in banner (default: auto-detected)
- `-v, --verbose`: Enable verbose logging

**Default Output Directory:**
- Windows IBA workstation: `C:\Users\IBA\RadiotherapyData\DICOM_exports`
- Other systems: `~/RadiotherapyData/DICOM_exports`

### Eclipse TPS Configuration

To send files from Eclipse to the DICOM listener:

1. **Open Eclipse** and go to External Beam Planning
2. **Configure DICOM Export** destination:
   - AE Title: `AUTOCONTOUR`
   - IP Address: Your workstation IP (displayed in the listener startup banner)
   - Port: `11112`
3. **Export CT and RTSTRUCT** by selecting the patient case and choosing "Export to DICOM"

**Note:** The listener will display your local IP address when it starts. Use this IP address in the Eclipse configuration.

### Received Data Organization

Files are automatically organized in the output directory by patient and study:

```
<output_dir>/
├── PatientID_001/
│   └── 20260201/
│       ├── CT_0001.dcm
│       ├── CT_0002.dcm
│       ├── ...
│       └── RS_label.dcm
├── PatientID_002/
│   └── 20260115/
│       ├── CT_0001.dcm
│       ├── ...
│       └── RS_ProstatePelvis.dcm
...
```

**File Naming Convention:**
- CT Images: `CT_XXXX.dcm` (where XXXX is the instance number)
- RT Structure Sets: `RS_<StructureSetLabel>.dcm`

### Statistics

When you stop the listener (Ctrl+C), it displays statistics:
- Total files received
- Number of CT images and RT structures
- Number of unique patients
- Uptime and error count

## API Service

Start the FastAPI service for production deployment:

```bash
python -m src.service.pipeline
```

API endpoints:
- `POST /contour` - Submit auto-contouring job
- `GET /task/{task_id}` - Check task status
- `GET /task/{task_id}/result` - Download RTSTRUCT
- `GET /health` - Service health check

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Formatting

```bash
black src/ scripts/
flake8 src/ scripts/
```

### Interactive Exploration

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Performance

Expected performance with 50-100 training cases:

| Structure | Dice Score | Training Time |
|-----------|-----------|---------------|
| Bladder | 0.93-0.96 | ~36 hours |
| Rectum | 0.91-0.94 | ~36 hours |
| Femoral Heads | 0.94-0.97 | ~36 hours |
| Bowel | 0.78-0.85 | ~36 hours |

*Training times for RTX 3090, 3d_fullres configuration, all 5 folds*

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 11+ GB VRAM (RTX 3080/A5000 or better)
- **RAM**: 32 GB minimum (64 GB recommended)
- **Storage**: 500 GB+ for datasets and models

### Software
- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or higher
- **OS**: Windows 10/11, Linux

## Citation

If you use this project, please cite:

```bibtex
@software{rt_contouring_ai,
  title = {RT Auto-Contouring by AI},
  author = {RT Contouring AI Team},
  year = {2026},
  url = {https://github.com/rezo935/RT-contouring-by-AI}
}
```

And the nnU-Net paper:

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - Deep learning framework
- [rt-utils](https://github.com/qurit/rt-utils) - DICOM RTSTRUCT utilities
- [SimpleITK](https://simpleitk.org/) - Medical image processing

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For questions and issues:
- 📫 Open an issue on GitHub
- 📖 Check the documentation
- 💬 Review existing issues and discussions

## Roadmap

- [ ] Support for additional body sites (head & neck, thorax)
- [ ] Multi-modality support (MRI, PET)
- [ ] Real-time inference optimization
- [ ] Integration with commercial TPS systems
- [ ] Uncertainty quantification
- [ ] Active learning for continuous improvement

---

**Project Status**: Active Development

**Last Updated**: February 2026
