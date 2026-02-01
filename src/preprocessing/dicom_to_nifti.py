"""DICOM to NIfTI conversion for radiotherapy data.

This module handles conversion of DICOM CT series and RTSTRUCT files
to NIfTI format suitable for nnU-Net training and inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder

from config.structures import (
    get_canonical_name,
    get_structure_label,
    get_all_canonical_names
)

logger = logging.getLogger(__name__)


class DicomToNiftiConverter:
    """
    Convert DICOM CT and RTSTRUCT to NIfTI format.
    
    This converter extracts CT images and structure masks from DICOM files
    and saves them in NIfTI format compatible with nnU-Net requirements.
    """
    
    def __init__(self):
        """Initialize converter."""
        pass
    
    def convert_case(
        self,
        case_id: str,
        ct_series_path: Path,
        rtstruct_path: Path,
        output_dir: Path,
        modality: str = "CT"
    ) -> Dict[str, Path]:
        """
        Convert a single case to NIfTI format.
        
        Args:
            case_id: Unique identifier for the case
            ct_series_path: Path to CT series directory
            rtstruct_path: Path to RTSTRUCT DICOM file
            output_dir: Directory to save NIfTI files
            modality: Imaging modality (default: "CT")
            
        Returns:
            Dictionary with paths to created files
            
        Example output structure for nnU-Net:
            - {output_dir}/{case_id}_0000.nii.gz (CT image)
            - {output_dir}/{case_id}.nii.gz (combined label mask)
        """
        logger.info(f"Converting case {case_id}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        try:
            # Load CT series
            ct_image = self._load_ct_series(ct_series_path)
            
            # Save CT as NIfTI (channel 0 for nnU-Net)
            ct_output_path = output_dir / f"{case_id}_0000.nii.gz"
            sitk.WriteImage(ct_image, str(ct_output_path))
            output_files["image"] = ct_output_path
            logger.info(f"Saved CT image to {ct_output_path}")
            
            # Load RTSTRUCT
            rtstruct = self._load_rtstruct(ct_series_path, rtstruct_path)
            
            # Extract structure masks
            combined_mask = self._extract_structure_masks(ct_image, rtstruct)
            
            # Save combined mask
            mask_output_path = output_dir / f"{case_id}.nii.gz"
            sitk.WriteImage(combined_mask, str(mask_output_path))
            output_files["label"] = mask_output_path
            logger.info(f"Saved label mask to {mask_output_path}")
            
        except Exception as e:
            logger.error(f"Error converting case {case_id}: {e}")
            raise
        
        return output_files
    
    def _load_ct_series(self, ct_series_path: Path) -> sitk.Image:
        """
        Load CT series using SimpleITK.
        
        Args:
            ct_series_path: Path to directory containing CT DICOM files
            
        Returns:
            SimpleITK Image object
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(ct_series_path))
        
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {ct_series_path}")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        logger.debug(f"Loaded CT series: size={image.GetSize()}, spacing={image.GetSpacing()}")
        
        return image
    
    def _load_rtstruct(self, ct_series_path: Path, rtstruct_path: Path) -> RTStructBuilder:
        """
        Load RTSTRUCT using rt-utils.
        
        Args:
            ct_series_path: Path to CT series directory
            rtstruct_path: Path to RTSTRUCT file
            
        Returns:
            RTStructBuilder object
        """
        try:
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(ct_series_path),
                rt_struct_path=str(rtstruct_path)
            )
            
            roi_names = rtstruct.get_roi_names()
            logger.debug(f"Loaded RTSTRUCT with {len(roi_names)} ROIs: {roi_names}")
            
            return rtstruct
        except Exception as e:
            logger.error(f"Error loading RTSTRUCT from {rtstruct_path}: {e}")
            raise
    
    def _extract_structure_masks(
        self,
        ct_image: sitk.Image,
        rtstruct: RTStructBuilder
    ) -> sitk.Image:
        """
        Extract structure masks from RTSTRUCT and combine into multi-label mask.
        
        Args:
            ct_image: CT image (used for spatial reference)
            rtstruct: RTSTRUCT containing structure contours
            
        Returns:
            Multi-label mask as SimpleITK Image
        """
        # Get image size from CT
        size = ct_image.GetSize()
        
        # Initialize combined mask (background = 0)
        combined_mask_array = np.zeros(size[::-1], dtype=np.uint8)  # Note: numpy uses (z, y, x)
        
        # Get all ROI names from RTSTRUCT
        roi_names = rtstruct.get_roi_names()
        
        # Process each structure
        structures_found = []
        
        for roi_name in roi_names:
            # Get canonical name
            canonical_name = get_canonical_name(roi_name)
            
            if canonical_name is None:
                logger.debug(f"Skipping unknown structure: {roi_name}")
                continue
            
            # Get label for this structure
            label = get_structure_label(roi_name)
            
            if label is None:
                logger.debug(f"No label defined for structure: {roi_name}")
                continue
            
            try:
                # Extract mask for this structure
                mask_3d = rtstruct.get_roi_mask_by_name(roi_name)
                mask_array = np.array(mask_3d, dtype=np.uint8)
                
                # Add to combined mask (later structures overwrite earlier ones if overlap)
                combined_mask_array[mask_array > 0] = label
                
                structures_found.append(canonical_name)
                logger.debug(f"Added structure {canonical_name} (label {label})")
                
            except Exception as e:
                logger.warning(f"Error extracting mask for {roi_name}: {e}")
        
        logger.info(f"Extracted {len(structures_found)} structures: {structures_found}")
        
        # Convert to SimpleITK image and copy spatial information from CT
        combined_mask = sitk.GetImageFromArray(combined_mask_array)
        combined_mask.CopyInformation(ct_image)
        
        return combined_mask
    
    def convert_batch(
        self,
        cases: List[Dict[str, any]],
        output_dir: Path,
        modality: str = "CT"
    ) -> Dict[str, Dict[str, Path]]:
        """
        Convert multiple cases to NIfTI format.
        
        Args:
            cases: List of case dictionaries with keys:
                   - case_id: Unique identifier
                   - ct_series_path: Path to CT series
                   - rtstruct_path: Path to RTSTRUCT
            output_dir: Directory to save NIfTI files
            modality: Imaging modality
            
        Returns:
            Dictionary mapping case_id to output files
        """
        results = {}
        
        for case in cases:
            case_id = case["case_id"]
            ct_series_path = Path(case["ct_series_path"])
            rtstruct_path = Path(case["rtstruct_path"])
            
            try:
                output_files = self.convert_case(
                    case_id=case_id,
                    ct_series_path=ct_series_path,
                    rtstruct_path=rtstruct_path,
                    output_dir=output_dir,
                    modality=modality
                )
                results[case_id] = output_files
            except Exception as e:
                logger.error(f"Failed to convert case {case_id}: {e}")
                results[case_id] = {"error": str(e)}
        
        logger.info(f"Successfully converted {len([r for r in results.values() if 'error' not in r])} / {len(cases)} cases")
        
        return results
    
    def extract_individual_masks(
        self,
        ct_image: sitk.Image,
        rtstruct: RTStructBuilder,
        output_dir: Path,
        case_id: str
    ) -> Dict[str, Path]:
        """
        Extract individual structure masks (useful for debugging or analysis).
        
        Args:
            ct_image: CT image
            rtstruct: RTSTRUCT
            output_dir: Output directory
            case_id: Case identifier
            
        Returns:
            Dictionary mapping structure names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        individual_masks = {}
        roi_names = rtstruct.get_roi_names()
        
        for roi_name in roi_names:
            canonical_name = get_canonical_name(roi_name)
            
            if canonical_name is None:
                continue
            
            try:
                # Extract mask
                mask_3d = rtstruct.get_roi_mask_by_name(roi_name)
                mask_array = np.array(mask_3d, dtype=np.uint8)
                
                # Convert to SimpleITK image
                mask_image = sitk.GetImageFromArray(mask_array)
                mask_image.CopyInformation(ct_image)
                
                # Save
                output_path = output_dir / f"{case_id}_{canonical_name}.nii.gz"
                sitk.WriteImage(mask_image, str(output_path))
                individual_masks[canonical_name] = output_path
                
            except Exception as e:
                logger.warning(f"Error extracting individual mask for {roi_name}: {e}")
        
        return individual_masks


def find_dicom_files(
    patient_dir: Path
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find CT series and RTSTRUCT files in a patient directory.
    
    Searches recursively through all subdirectories to support both:
    - Flat structure: PatientID/file.dcm
    - Nested structure: PatientID/Date/file.dcm
    
    If multiple CT series directories exist, returns the first one found.
    This is appropriate for typical RT planning workflows where one CT 
    series is used per patient study.
    
    Args:
        patient_dir: Directory containing patient DICOM files
        
    Returns:
        Tuple of (ct_series_path, rtstruct_path)
        - ct_series_path: Directory containing CT DICOM files
        - rtstruct_path: Path to RTSTRUCT DICOM file
    """
    patient_dir = Path(patient_dir)
    
    ct_series_path = None
    rtstruct_path = None
    
    # Find all DICOM files recursively
    dicom_files = list(patient_dir.glob("**/*.dcm"))
    
    if not dicom_files:
        return ct_series_path, rtstruct_path
    
    # Group files by their parent directory
    files_by_dir = {}
    for file_path in dicom_files:
        parent_dir = file_path.parent
        if parent_dir not in files_by_dir:
            files_by_dir[parent_dir] = []
        files_by_dir[parent_dir].append(file_path)
    
    # Process each directory to identify CT series and RTSTRUCT
    for dir_path, files in files_by_dir.items():
        for file_path in files:
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # Check for CT modality
                if hasattr(dcm, "Modality") and dcm.Modality == "CT":
                    if ct_series_path is None:
                        ct_series_path = dir_path
                
                # Check for RTSTRUCT by modality or SOP Class UID
                is_rtstruct = False
                if hasattr(dcm, "Modality") and dcm.Modality == "RTSTRUCT":
                    is_rtstruct = True
                if hasattr(dcm, "SOPClassUID") and dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3":
                    is_rtstruct = True
                
                if is_rtstruct and rtstruct_path is None:
                    rtstruct_path = file_path
                
                # Stop early if we found both
                if ct_series_path is not None and rtstruct_path is not None:
                    return ct_series_path, rtstruct_path
                    
            except Exception as e:
                logger.debug(f"Error reading {file_path}: {e}")
                continue
    
    return ct_series_path, rtstruct_path
