"""Convert NIfTI segmentation masks back to DICOM RTSTRUCT.

This module handles conversion of nnU-Net predictions (NIfTI format)
back to DICOM RTSTRUCT for import into treatment planning systems.
"""

from pathlib import Path
from typing import Dict, Optional, List
import logging

import numpy as np
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder

from config.structures import (
    get_all_canonical_names,
    get_structure_color,
    get_all_labels
)

logger = logging.getLogger(__name__)


class NiftiToRtstructConverter:
    """
    Convert NIfTI segmentation masks to DICOM RTSTRUCT.
    
    This converter creates DICOM RTSTRUCT files from nnU-Net predictions,
    maintaining proper DICOM references and setting structure colors.
    """
    
    def __init__(self):
        """Initialize converter."""
        pass
    
    def convert(
        self,
        segmentation_path: Path,
        ct_series_path: Path,
        output_path: Path,
        series_description: str = "AI Auto-Contours",
        structure_set_label: str = "AI_Segmentation",
        structure_set_name: str = "nnUNet Pelvis OAR"
    ) -> Path:
        """
        Convert NIfTI segmentation to DICOM RTSTRUCT.
        
        Args:
            segmentation_path: Path to segmentation NIfTI file
            ct_series_path: Path to original CT DICOM series
            output_path: Path to save RTSTRUCT file
            series_description: Series description for RTSTRUCT
            structure_set_label: Structure set label
            structure_set_name: Structure set name
            
        Returns:
            Path to created RTSTRUCT file
        """
        logger.info(f"Converting segmentation to RTSTRUCT")
        logger.info(f"  Segmentation: {segmentation_path}")
        logger.info(f"  CT series: {ct_series_path}")
        logger.info(f"  Output: {output_path}")
        
        # Load segmentation
        segmentation = sitk.ReadImage(str(segmentation_path))
        seg_array = sitk.GetArrayFromImage(segmentation)
        
        # Create RTSTRUCT
        logger.info("Creating RTSTRUCT from CT series...")
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_series_path))
        
        # Set metadata
        rtstruct.ds.SeriesDescription = series_description
        rtstruct.ds.StructureSetLabel = structure_set_label
        rtstruct.ds.StructureSetName = structure_set_name
        
        # Get label mapping
        label_mapping = get_all_labels()
        
        # Add each structure
        structures_added = []
        
        for structure_name, label in label_mapping.items():
            # Extract binary mask for this structure
            structure_mask = (seg_array == label).astype(bool)
            
            # Check if structure is present
            if not np.any(structure_mask):
                logger.debug(f"Structure {structure_name} not found in segmentation")
                continue
            
            # Get color
            color = get_structure_color(structure_name)
            if color is None:
                color = (255, 0, 0)  # Default to red
            
            try:
                # Add ROI to RTSTRUCT
                logger.info(f"Adding structure: {structure_name} (label {label})")
                
                rtstruct.add_roi(
                    mask=structure_mask,
                    color=color,
                    name=structure_name
                )
                
                structures_added.append(structure_name)
                
            except Exception as e:
                logger.warning(f"Error adding structure {structure_name}: {e}")
        
        if len(structures_added) == 0:
            logger.warning("No structures were added to RTSTRUCT")
        else:
            logger.info(f"Added {len(structures_added)} structures: {structures_added}")
        
        # Save RTSTRUCT
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rtstruct.save(str(output_path))
        logger.info(f"Saved RTSTRUCT to {output_path}")
        
        return output_path
    
    def convert_batch(
        self,
        cases: List[Dict[str, Path]],
        output_dir: Path,
        series_description: str = "AI Auto-Contours"
    ) -> Dict[str, Path]:
        """
        Convert multiple segmentations to RTSTRUCT.
        
        Args:
            cases: List of case dictionaries with keys:
                   - case_id: Unique identifier
                   - segmentation_path: Path to segmentation NIfTI
                   - ct_series_path: Path to CT DICOM series
            output_dir: Directory to save RTSTRUCT files
            series_description: Series description
            
        Returns:
            Dictionary mapping case_id to RTSTRUCT file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for case in cases:
            case_id = case["case_id"]
            
            try:
                output_path = output_dir / f"{case_id}_RTSTRUCT.dcm"
                
                rtstruct_path = self.convert(
                    segmentation_path=case["segmentation_path"],
                    ct_series_path=case["ct_series_path"],
                    output_path=output_path,
                    series_description=series_description
                )
                
                results[case_id] = rtstruct_path
                
            except Exception as e:
                logger.error(f"Error converting case {case_id}: {e}")
                results[case_id] = None
        
        logger.info(f"Converted {len([r for r in results.values() if r is not None])} / {len(cases)} cases")
        
        return results
    
    def convert_individual_structures(
        self,
        segmentation_path: Path,
        ct_series_path: Path,
        output_dir: Path,
        structures: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Create separate RTSTRUCT files for each structure.
        
        Args:
            segmentation_path: Path to segmentation NIfTI
            ct_series_path: Path to CT series
            output_dir: Output directory
            structures: List of structure names to export (None for all)
            
        Returns:
            Dictionary mapping structure names to RTSTRUCT paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load segmentation
        segmentation = sitk.ReadImage(str(segmentation_path))
        seg_array = sitk.GetArrayFromImage(segmentation)
        
        # Get label mapping
        label_mapping = get_all_labels()
        
        if structures is None:
            structures = get_all_canonical_names()
        
        results = {}
        
        for structure_name in structures:
            if structure_name not in label_mapping:
                logger.warning(f"Unknown structure: {structure_name}")
                continue
            
            label = label_mapping[structure_name]
            
            # Extract binary mask
            structure_mask = (seg_array == label).astype(bool)
            
            if not np.any(structure_mask):
                logger.debug(f"Structure {structure_name} not found in segmentation")
                continue
            
            try:
                # Create RTSTRUCT for this structure
                rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_series_path))
                
                color = get_structure_color(structure_name)
                if color is None:
                    color = (255, 0, 0)
                
                rtstruct.add_roi(
                    mask=structure_mask,
                    color=color,
                    name=structure_name
                )
                
                # Save
                output_path = output_dir / f"{structure_name}_RTSTRUCT.dcm"
                rtstruct.save(str(output_path))
                
                results[structure_name] = output_path
                logger.info(f"Saved {structure_name} to {output_path}")
                
            except Exception as e:
                logger.warning(f"Error creating RTSTRUCT for {structure_name}: {e}")
                results[structure_name] = None
        
        return results


def merge_rtstructs(
    rtstruct_paths: List[Path],
    ct_series_path: Path,
    output_path: Path,
    series_description: str = "Merged Auto-Contours"
) -> Path:
    """
    Merge multiple RTSTRUCT files into one.
    
    Args:
        rtstruct_paths: List of RTSTRUCT file paths to merge
        ct_series_path: Path to CT series
        output_path: Path to save merged RTSTRUCT
        series_description: Series description
        
    Returns:
        Path to merged RTSTRUCT file
    """
    logger.info(f"Merging {len(rtstruct_paths)} RTSTRUCT files")
    
    # Create new RTSTRUCT
    merged_rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_series_path))
    merged_rtstruct.ds.SeriesDescription = series_description
    
    # Add ROIs from each RTSTRUCT
    for rtstruct_path in rtstruct_paths:
        try:
            # Load RTSTRUCT
            source_rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(ct_series_path),
                rt_struct_path=str(rtstruct_path)
            )
            
            # Get all ROI names
            roi_names = source_rtstruct.get_roi_names()
            
            for roi_name in roi_names:
                # Get mask
                mask = source_rtstruct.get_roi_mask_by_name(roi_name)
                
                # Get color (try to preserve original)
                # This requires accessing the DICOM dataset directly
                color = (255, 0, 0)  # Default
                
                # Add to merged RTSTRUCT
                merged_rtstruct.add_roi(
                    mask=mask,
                    color=color,
                    name=roi_name
                )
                
                logger.debug(f"Added ROI: {roi_name}")
                
        except Exception as e:
            logger.warning(f"Error processing {rtstruct_path}: {e}")
    
    # Save merged RTSTRUCT
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_rtstruct.save(str(output_path))
    
    logger.info(f"Saved merged RTSTRUCT to {output_path}")
    
    return output_path


def validate_rtstruct(rtstruct_path: Path, ct_series_path: Path) -> bool:
    """
    Validate RTSTRUCT file.
    
    Args:
        rtstruct_path: Path to RTSTRUCT file
        ct_series_path: Path to CT series
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to load RTSTRUCT
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_series_path),
            rt_struct_path=str(rtstruct_path)
        )
        
        # Check if it has ROIs
        roi_names = rtstruct.get_roi_names()
        
        if len(roi_names) == 0:
            logger.warning("RTSTRUCT has no ROIs")
            return False
        
        logger.info(f"RTSTRUCT is valid with {len(roi_names)} ROIs: {roi_names}")
        return True
        
    except Exception as e:
        logger.error(f"RTSTRUCT validation failed: {e}")
        return False
