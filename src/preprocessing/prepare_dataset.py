"""Prepare datasets for nnU-Net training.

This module creates the dataset.json file and sets up the proper directory
structure required by nnU-Net v2.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import shutil

import numpy as np

from config.structures import get_all_canonical_names, get_all_labels
from config.paths import PathConfig

logger = logging.getLogger(__name__)


class NnUNetDatasetPreparator:
    """
    Prepare dataset for nnU-Net training.
    
    This class handles creation of dataset.json, directory structure setup,
    and train/validation split creation.
    """
    
    def __init__(self, path_config: PathConfig):
        """
        Initialize dataset preparator.
        
        Args:
            path_config: Path configuration object
        """
        self.path_config = path_config
    
    def create_dataset(
        self,
        dataset_id: int,
        dataset_name: str,
        nifti_source_dir: Path,
        description: str = "Pelvis OAR Segmentation",
        modality: str = "CT",
        train_val_split: float = 0.8
    ) -> Path:
        """
        Create nnU-Net dataset with proper structure and metadata.
        
        Args:
            dataset_id: Dataset ID (e.g., 001)
            dataset_name: Dataset name (e.g., "PelvisOAR")
            nifti_source_dir: Directory containing converted NIfTI files
            description: Dataset description
            modality: Imaging modality
            train_val_split: Fraction of data to use for training (rest for validation)
            
        Returns:
            Path to created dataset directory
        """
        logger.info(f"Creating nnU-Net dataset: Dataset{dataset_id:03d}_{dataset_name}")
        
        # Create dataset directory structure
        dataset_path = self.path_config.get_dataset_path(dataset_id, dataset_name)
        images_path = self.path_config.get_dataset_images_path(dataset_id, dataset_name)
        labels_path = self.path_config.get_dataset_labels_path(dataset_id, dataset_name)
        
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Find all NIfTI files in source directory
        image_files = sorted(nifti_source_dir.glob("*_0000.nii.gz"))
        label_files = sorted(nifti_source_dir.glob("*.nii.gz"))
        label_files = [f for f in label_files if not f.name.endswith("_0000.nii.gz")]
        
        logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")
        
        # Copy files to dataset directories
        for img_file in image_files:
            shutil.copy2(img_file, images_path / img_file.name)
        
        for lbl_file in label_files:
            shutil.copy2(lbl_file, labels_path / lbl_file.name)
        
        # Create dataset.json
        dataset_json = self._create_dataset_json(
            dataset_name=dataset_name,
            description=description,
            modality=modality,
            num_training=len(image_files)
        )
        
        json_path = dataset_path / "dataset.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        logger.info(f"Created dataset.json at {json_path}")
        
        # Create train/val split
        self._create_splits_final(dataset_path, len(image_files), train_val_split)
        
        logger.info(f"Dataset created successfully at {dataset_path}")
        
        return dataset_path
    
    def _create_dataset_json(
        self,
        dataset_name: str,
        description: str,
        modality: str,
        num_training: int
    ) -> Dict:
        """
        Create dataset.json content for nnU-Net.
        
        Args:
            dataset_name: Name of the dataset
            description: Dataset description
            modality: Imaging modality
            num_training: Number of training cases
            
        Returns:
            Dictionary with dataset.json content
        """
        # Get label mapping
        labels_dict = get_all_labels()
        
        # Create labels dict with background
        labels_for_json = {"background": 0}
        labels_for_json.update(labels_dict)
        
        # Create channel names (nnU-Net expects dict with integer keys as strings)
        channel_names = {
            "0": modality
        }
        
        # File ending
        file_ending = ".nii.gz"
        
        dataset_json = {
            "name": dataset_name,
            "description": description,
            "reference": "RT Contouring AI Project",
            "licence": "Internal Use",
            "release": "1.0",
            "tensorImageSize": "4D",
            "modality": channel_names,
            "labels": labels_for_json,
            "numTraining": num_training,
            "numTest": 0,
            "file_ending": file_ending,
        }
        
        return dataset_json
    
    def _create_splits_final(
        self,
        dataset_path: Path,
        num_cases: int,
        train_val_split: float
    ) -> None:
        """
        Create splits_final.json for train/validation split.
        
        Args:
            dataset_path: Path to dataset directory
            num_cases: Total number of cases
            train_val_split: Fraction for training
        """
        # Get case identifiers (assumes files are named consistently)
        images_path = dataset_path / "imagesTr"
        image_files = sorted(images_path.glob("*_0000.nii.gz"))
        
        case_ids = [f.name.replace("_0000.nii.gz", "") for f in image_files]
        
        # Create train/val split
        num_train = int(len(case_ids) * train_val_split)
        
        train_ids = case_ids[:num_train]
        val_ids = case_ids[num_train:]
        
        # Create splits for 5-fold cross-validation (nnU-Net default)
        splits = []
        
        for fold in range(5):
            # Distribute validation cases across folds
            fold_size = len(val_ids) // 5
            fold_start = fold * fold_size
            fold_end = fold_start + fold_size if fold < 4 else len(val_ids)
            
            fold_val = val_ids[fold_start:fold_end]
            fold_train = [cid for cid in case_ids if cid not in fold_val]
            
            splits.append({
                "train": fold_train,
                "val": fold_val
            })
        
        # Save splits_final.json
        splits_path = dataset_path / "splits_final.json"
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Created splits_final.json with {len(train_ids)} train, {len(val_ids)} val cases")
    
    def verify_dataset(self, dataset_id: int, dataset_name: str) -> bool:
        """
        Verify dataset integrity.
        
        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
            
        Returns:
            True if dataset is valid, False otherwise
        """
        dataset_path = self.path_config.get_dataset_path(dataset_id, dataset_name)
        
        # Check directory exists
        if not dataset_path.exists():
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            return False
        
        # Check dataset.json exists
        json_path = dataset_path / "dataset.json"
        if not json_path.exists():
            logger.error(f"dataset.json not found: {json_path}")
            return False
        
        # Check imagesTr and labelsTr directories
        images_path = dataset_path / "imagesTr"
        labels_path = dataset_path / "labelsTr"
        
        if not images_path.exists():
            logger.error(f"imagesTr directory not found: {images_path}")
            return False
        
        if not labels_path.exists():
            logger.error(f"labelsTr directory not found: {labels_path}")
            return False
        
        # Check matching files
        image_files = list(images_path.glob("*_0000.nii.gz"))
        label_files = list(labels_path.glob("*.nii.gz"))
        
        if len(image_files) == 0:
            logger.error("No image files found")
            return False
        
        if len(label_files) == 0:
            logger.error("No label files found")
            return False
        
        # Check each image has corresponding label
        for img_file in image_files:
            case_id = img_file.name.replace("_0000.nii.gz", "")
            label_file = labels_path / f"{case_id}.nii.gz"
            
            if not label_file.exists():
                logger.error(f"Missing label for {case_id}")
                return False
        
        logger.info(f"Dataset verification passed: {len(image_files)} cases")
        return True
    
    def get_dataset_statistics(
        self,
        dataset_id: int,
        dataset_name: str
    ) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset_path = self.path_config.get_dataset_path(dataset_id, dataset_name)
        
        stats = {
            "dataset_path": str(dataset_path),
            "exists": dataset_path.exists()
        }
        
        if not dataset_path.exists():
            return stats
        
        # Count files
        images_path = dataset_path / "imagesTr"
        labels_path = dataset_path / "labelsTr"
        
        if images_path.exists():
            stats["num_images"] = len(list(images_path.glob("*_0000.nii.gz")))
        
        if labels_path.exists():
            stats["num_labels"] = len(list(labels_path.glob("*.nii.gz")))
        
        # Check dataset.json
        json_path = dataset_path / "dataset.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                dataset_json = json.load(f)
            stats["dataset_json"] = dataset_json
        
        # Check splits
        splits_path = dataset_path / "splits_final.json"
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            stats["num_folds"] = len(splits)
            if len(splits) > 0:
                stats["fold_0_train"] = len(splits[0]["train"])
                stats["fold_0_val"] = len(splits[0]["val"])
        
        return stats


def prepare_dataset_from_nifti(
    nifti_dir: Path,
    dataset_id: int,
    dataset_name: str,
    path_config: Optional[PathConfig] = None,
    description: str = "Pelvis OAR Segmentation",
    modality: str = "CT",
    train_val_split: float = 0.8
) -> Path:
    """
    Convenience function to prepare nnU-Net dataset from NIfTI files.
    
    Args:
        nifti_dir: Directory containing NIfTI files
        dataset_id: Dataset ID
        dataset_name: Dataset name
        path_config: Optional path configuration (creates default if None)
        description: Dataset description
        modality: Imaging modality
        train_val_split: Train/validation split ratio
        
    Returns:
        Path to created dataset
    """
    if path_config is None:
        from config.paths import get_path_config
        path_config = get_path_config()
    
    preparator = NnUNetDatasetPreparator(path_config)
    
    dataset_path = preparator.create_dataset(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        nifti_source_dir=nifti_dir,
        description=description,
        modality=modality,
        train_val_split=train_val_split
    )
    
    # Verify dataset
    if not preparator.verify_dataset(dataset_id, dataset_name):
        raise ValueError("Dataset verification failed")
    
    return dataset_path
