"""Setup script for RT-contouring-by-AI package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="rt-contouring-ai",
    version="0.1.0",
    author="RT Contouring AI Team",
    author_email="",
    description="Radiotherapy auto-contouring using AI - Pelvis OAR segmentation with nnU-Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rezo935/RT-contouring-by-AI",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "nnunetv2>=2.2",
        "SimpleITK>=2.2.0",
        "pydicom>=2.4.0",
        "rt-utils>=1.2.7",
        "nibabel>=5.1.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rt-quality-assessment=scripts.run_quality_assessment:main",
            "rt-preprocessing=scripts.run_preprocessing:main",
            "rt-training=scripts.run_training:main",
            "rt-inference=scripts.run_inference:main",
        ],
    },
)
