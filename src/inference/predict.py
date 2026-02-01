"""Inference module for running predictions on new cases.

This module handles the complete inference pipeline from DICOM input
to segmentation output.
"""

from pathlib import Path
from typing import Dict, Optional, List
import logging
import tempfile
import shutil

import SimpleITK as sitk
import numpy as np

from config.paths import PathConfig
from src.preprocessing.dicom_to_nifti import DicomToNiftiConverter, find_dicom_files
from src.training.train_nnunet import NnUNetTrainer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Inference engine for running predictions on new cases.
    
    This class handles the complete pipeline:
    1. Load DICOM CT
    2. Convert to NIfTI
    3. Run nnU-Net inference
    4. Return segmentation masks
    """
    
    def __init__(
        self,
        dataset_id: int,
        path_config: PathConfig,
        configuration: str = "3d_fullres",
        trainer: str = "nnUNetTrainer",
        plans: str = "nnUNetPlans",
        folds: str = "all"
    ):
        """
        Initialize inference engine.
        
        Args:
            dataset_id: Trained dataset ID
            path_config: Path configuration
            configuration: Model configuration
            trainer: Trainer class name
            plans: Plans identifier
            folds: Which folds to use for ensemble
        """
        self.dataset_id = dataset_id
        self.path_config = path_config
        self.configuration = configuration
        self.trainer = trainer
        self.plans = plans
        self.folds = folds
        
        self.converter = DicomToNiftiConverter()
        self.nnunet_trainer = NnUNetTrainer(path_config)
        
        # Verify model exists
        self._verify_model()
    
    def _verify_model(self) -> None:
        """Verify that trained model exists."""
        status = self.nnunet_trainer.get_training_status(
            dataset_id=self.dataset_id,
            configuration=self.configuration,
            trainer=self.trainer,
            plans=self.plans
        )
        
        if not status["exists"]:
            raise ValueError(f"Model not found for dataset {self.dataset_id}")
        
        if len(status["folds_trained"]) == 0:
            raise ValueError(f"No trained folds found for dataset {self.dataset_id}")
        
        logger.info(f"Model verified: {len(status['folds_trained'])} folds available")
    
    def predict_from_dicom(
        self,
        ct_series_path: Path,
        output_dir: Optional[Path] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run inference on a DICOM CT series.
        
        Args:
            ct_series_path: Path to CT series directory
            output_dir: Output directory (uses default if None)
            case_id: Case identifier (auto-generated if None)
            
        Returns:
            Dictionary with paths to output files
        """
        if case_id is None:
            case_id = f"case_{Path(ct_series_path).name}"
        
        if output_dir is None:
            output_dir = self.path_config.inference_output / case_id
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running inference on case: {case_id}")
        
        # Create temporary directories for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_nifti_dir = temp_path / "input"
            output_nifti_dir = temp_path / "output"
            
            input_nifti_dir.mkdir(parents=True, exist_ok=True)
            output_nifti_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert DICOM to NIfTI
            logger.info("Converting DICOM to NIfTI...")
            ct_image = self.converter._load_ct_series(ct_series_path)
            
            # Save as NIfTI in nnU-Net format
            nifti_path = input_nifti_dir / f"{case_id}_0000.nii.gz"
            sitk.WriteImage(ct_image, str(nifti_path))
            logger.info(f"Saved input NIfTI to {nifti_path}")
            
            # Run inference
            logger.info("Running nnU-Net inference...")
            success = self.nnunet_trainer.predict(
                input_folder=input_nifti_dir,
                output_folder=output_nifti_dir,
                dataset_id=self.dataset_id,
                configuration=self.configuration,
                trainer=self.trainer,
                plans=self.plans,
                folds=self.folds
            )
            
            if not success:
                raise RuntimeError("Inference failed")
            
            # Copy results to output directory
            prediction_file = output_nifti_dir / f"{case_id}.nii.gz"
            
            if not prediction_file.exists():
                raise RuntimeError(f"Prediction file not found: {prediction_file}")
            
            output_prediction = output_dir / f"{case_id}_segmentation.nii.gz"
            shutil.copy2(prediction_file, output_prediction)
            
            logger.info(f"Saved prediction to {output_prediction}")
            
            # Also copy the input CT for reference
            output_ct = output_dir / f"{case_id}_ct.nii.gz"
            shutil.copy2(nifti_path, output_ct)
            
            return {
                "case_id": case_id,
                "ct": output_ct,
                "segmentation": output_prediction,
                "output_dir": output_dir
            }
    
    def predict_from_nifti(
        self,
        ct_nifti_path: Path,
        output_dir: Optional[Path] = None,
        case_id: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run inference on a NIfTI CT image.
        
        Args:
            ct_nifti_path: Path to CT NIfTI file
            output_dir: Output directory
            case_id: Case identifier
            
        Returns:
            Dictionary with paths to output files
        """
        if case_id is None:
            case_id = ct_nifti_path.stem.replace("_0000", "")
        
        if output_dir is None:
            output_dir = self.path_config.inference_output / case_id
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running inference on case: {case_id}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_nifti_dir = temp_path / "input"
            output_nifti_dir = temp_path / "output"
            
            input_nifti_dir.mkdir(parents=True, exist_ok=True)
            output_nifti_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy input to temporary directory with correct naming
            input_file = input_nifti_dir / f"{case_id}_0000.nii.gz"
            shutil.copy2(ct_nifti_path, input_file)
            
            # Run inference
            logger.info("Running nnU-Net inference...")
            success = self.nnunet_trainer.predict(
                input_folder=input_nifti_dir,
                output_folder=output_nifti_dir,
                dataset_id=self.dataset_id,
                configuration=self.configuration,
                trainer=self.trainer,
                plans=self.plans,
                folds=self.folds
            )
            
            if not success:
                raise RuntimeError("Inference failed")
            
            # Copy results to output directory
            prediction_file = output_nifti_dir / f"{case_id}.nii.gz"
            
            if not prediction_file.exists():
                raise RuntimeError(f"Prediction file not found: {prediction_file}")
            
            output_prediction = output_dir / f"{case_id}_segmentation.nii.gz"
            shutil.copy2(prediction_file, output_prediction)
            
            logger.info(f"Saved prediction to {output_prediction}")
            
            return {
                "case_id": case_id,
                "segmentation": output_prediction,
                "output_dir": output_dir
            }
    
    def batch_predict(
        self,
        input_cases: List[Dict[str, Path]],
        output_dir: Path
    ) -> List[Dict[str, Path]]:
        """
        Run inference on multiple cases.
        
        Args:
            input_cases: List of case dictionaries with keys:
                        - "case_id": Unique identifier
                        - "ct_series_path": Path to CT series (for DICOM)
                        OR
                        - "ct_nifti_path": Path to CT NIfTI (for NIfTI)
            output_dir: Output directory for all results
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for case_info in input_cases:
            case_id = case_info["case_id"]
            case_output_dir = output_dir / case_id
            
            try:
                if "ct_series_path" in case_info:
                    result = self.predict_from_dicom(
                        ct_series_path=case_info["ct_series_path"],
                        output_dir=case_output_dir,
                        case_id=case_id
                    )
                elif "ct_nifti_path" in case_info:
                    result = self.predict_from_nifti(
                        ct_nifti_path=case_info["ct_nifti_path"],
                        output_dir=case_output_dir,
                        case_id=case_id
                    )
                else:
                    raise ValueError("Case must have either 'ct_series_path' or 'ct_nifti_path'")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing case {case_id}: {e}")
                results.append({
                    "case_id": case_id,
                    "error": str(e)
                })
        
        logger.info(f"Batch inference completed: {len([r for r in results if 'error' not in r])} / {len(input_cases)} successful")
        
        return results


def load_segmentation(segmentation_path: Path) -> sitk.Image:
    """
    Load segmentation NIfTI file.
    
    Args:
        segmentation_path: Path to segmentation file
        
    Returns:
        SimpleITK Image object
    """
    return sitk.ReadImage(str(segmentation_path))


def extract_structure_mask(
    segmentation: sitk.Image,
    label: int
) -> sitk.Image:
    """
    Extract binary mask for a specific structure from multi-label segmentation.
    
    Args:
        segmentation: Multi-label segmentation image
        label: Label value to extract
        
    Returns:
        Binary mask for the specified label
    """
    # Convert to numpy array
    seg_array = sitk.GetArrayFromImage(segmentation)
    
    # Create binary mask
    mask_array = (seg_array == label).astype(np.uint8)
    
    # Convert back to SimpleITK image
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(segmentation)
    
    return mask


def get_structure_volume(
    segmentation: sitk.Image,
    label: int
) -> float:
    """
    Calculate volume of a structure in cm³.
    
    Args:
        segmentation: Multi-label segmentation image
        label: Label value
        
    Returns:
        Volume in cm³
    """
    seg_array = sitk.GetArrayFromImage(segmentation)
    
    num_voxels = np.sum(seg_array == label)
    
    spacing = segmentation.GetSpacing()
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_volume_cc = voxel_volume_mm3 / 1000.0
    
    volume_cc = num_voxels * voxel_volume_cc
    
    return volume_cc


def get_all_structure_volumes(
    segmentation: sitk.Image,
    label_mapping: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate volumes for all structures.
    
    Args:
        segmentation: Multi-label segmentation image
        label_mapping: Dictionary mapping structure names to labels
        
    Returns:
        Dictionary mapping structure names to volumes in cm³
    """
    volumes = {}
    
    for structure_name, label in label_mapping.items():
        volume = get_structure_volume(segmentation, label)
        volumes[structure_name] = volume
    
    return volumes
