"""Path configuration helper for the RT contouring project.

This module provides utilities for managing paths in a cross-platform manner,
with special attention to Windows compatibility.
"""

from pathlib import Path
from typing import Optional
import os


class PathConfig:
    """
    Configuration class for managing project paths.
    
    This class provides centralized path management for data directories,
    nnU-Net paths, and output directories. All paths use pathlib for
    cross-platform compatibility.
    """
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        data_root: Optional[Path] = None,
        nnunet_root: Optional[Path] = None
    ):
        """
        Initialize path configuration.
        
        Args:
            project_root: Root directory of the project (defaults to parent of this file)
            data_root: Root directory for data storage (defaults to project_root/data)
            nnunet_root: Root directory for nnU-Net data (defaults to data_root/nnUNet)
        """
        # Project root
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.absolute()
        else:
            self.project_root = Path(project_root).absolute()
        
        # Data root
        if data_root is None:
            self.data_root = self.project_root / "data"
        else:
            self.data_root = Path(data_root).absolute()
        
        # nnU-Net root (can be overridden by environment variables)
        if nnunet_root is None:
            env_nnunet = os.environ.get("nnUNet_raw")
            if env_nnunet:
                self.nnunet_root = Path(env_nnunet).parent.absolute()
            else:
                self.nnunet_root = self.data_root / "nnUNet"
        else:
            self.nnunet_root = Path(nnunet_root).absolute()
    
    # DICOM data paths
    @property
    def dicom_raw(self) -> Path:
        """Path to raw DICOM data."""
        return self.data_root / "dicom_raw"
    
    @property
    def dicom_processed(self) -> Path:
        """Path to processed DICOM data."""
        return self.data_root / "dicom_processed"
    
    # Quality assessment paths
    @property
    def quality_reports(self) -> Path:
        """Path to quality assessment reports."""
        return self.data_root / "quality_reports"
    
    # nnU-Net paths (following nnU-Net v2 conventions)
    @property
    def nnunet_raw(self) -> Path:
        """Path to nnU-Net raw data directory."""
        return self.nnunet_root / "nnUNet_raw"
    
    @property
    def nnunet_preprocessed(self) -> Path:
        """Path to nnU-Net preprocessed data directory."""
        return self.nnunet_root / "nnUNet_preprocessed"
    
    @property
    def nnunet_results(self) -> Path:
        """Path to nnU-Net training results directory."""
        return self.nnunet_root / "nnUNet_results"
    
    def get_dataset_path(self, dataset_id: int, dataset_name: str) -> Path:
        """
        Get path to a specific nnU-Net dataset.
        
        Args:
            dataset_id: Dataset ID (e.g., 001)
            dataset_name: Dataset name (e.g., "PelvisOAR")
            
        Returns:
            Path to dataset directory
        """
        dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
        return self.nnunet_raw / dataset_folder
    
    def get_dataset_images_path(self, dataset_id: int, dataset_name: str) -> Path:
        """
        Get path to images directory for a specific dataset.
        
        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
            
        Returns:
            Path to imagesTr directory
        """
        return self.get_dataset_path(dataset_id, dataset_name) / "imagesTr"
    
    def get_dataset_labels_path(self, dataset_id: int, dataset_name: str) -> Path:
        """
        Get path to labels directory for a specific dataset.
        
        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
            
        Returns:
            Path to labelsTr directory
        """
        return self.get_dataset_path(dataset_id, dataset_name) / "labelsTr"
    
    # Output paths
    @property
    def inference_output(self) -> Path:
        """Path to inference output directory."""
        return self.data_root / "inference_output"
    
    @property
    def temp_dir(self) -> Path:
        """Path to temporary directory."""
        return self.data_root / "temp"
    
    # Logs and reports
    @property
    def logs(self) -> Path:
        """Path to logs directory."""
        return self.project_root / "logs"
    
    def create_directories(self) -> None:
        """
        Create all necessary directories if they don't exist.
        """
        directories = [
            self.data_root,
            self.dicom_raw,
            self.dicom_processed,
            self.quality_reports,
            self.nnunet_raw,
            self.nnunet_preprocessed,
            self.nnunet_results,
            self.inference_output,
            self.temp_dir,
            self.logs
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_nnunet_env(self) -> None:
        """
        Set up nnU-Net environment variables.
        
        This sets the required environment variables for nnU-Net v2:
        - nnUNet_raw
        - nnUNet_preprocessed
        - nnUNet_results
        """
        os.environ["nnUNet_raw"] = str(self.nnunet_raw)
        os.environ["nnUNet_preprocessed"] = str(self.nnunet_preprocessed)
        os.environ["nnUNet_results"] = str(self.nnunet_results)
        
        # Create directories
        self.nnunet_raw.mkdir(parents=True, exist_ok=True)
        self.nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
        self.nnunet_results.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        """String representation of PathConfig."""
        return (
            f"PathConfig(\n"
            f"  project_root={self.project_root},\n"
            f"  data_root={self.data_root},\n"
            f"  nnunet_root={self.nnunet_root}\n"
            f")"
        )


# Global default instance
_default_config: Optional[PathConfig] = None


def get_path_config(
    project_root: Optional[Path] = None,
    data_root: Optional[Path] = None,
    nnunet_root: Optional[Path] = None,
    force_new: bool = False
) -> PathConfig:
    """
    Get or create the global PathConfig instance.
    
    Args:
        project_root: Optional project root path
        data_root: Optional data root path
        nnunet_root: Optional nnU-Net root path
        force_new: If True, create a new instance even if one exists
        
    Returns:
        PathConfig instance
    """
    global _default_config
    
    if _default_config is None or force_new:
        _default_config = PathConfig(
            project_root=project_root,
            data_root=data_root,
            nnunet_root=nnunet_root
        )
    
    return _default_config
