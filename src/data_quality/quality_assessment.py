"""Quality assessment for radiotherapy contouring datasets.

This module provides tools for assessing the quality of DICOM CT and RTSTRUCT files,
including volume checks, outlier detection, and quality reporting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from rt_utils import RTStructBuilder
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import seaborn as sns

from config.structures import (
    get_canonical_name,
    get_volume_range,
    get_all_canonical_names,
    STRUCTURE_LABELS
)

logger = logging.getLogger(__name__)


class StructureMetrics:
    """Container for structure metrics."""
    
    def __init__(
        self,
        name: str,
        volume_cc: float,
        centroid: Tuple[float, float, float],
        bounding_box: Tuple[int, int, int, int, int, int],
        num_components: int,
        sphericity: float,
        present: bool = True
    ):
        self.name = name
        self.volume_cc = volume_cc
        self.centroid = centroid
        self.bounding_box = bounding_box
        self.num_components = num_components
        self.sphericity = sphericity
        self.present = present


class QualityIssue:
    """Container for quality issues."""
    
    def __init__(
        self,
        severity: str,  # "error", "warning", "info"
        structure: Optional[str],
        message: str
    ):
        self.severity = severity
        self.structure = structure
        self.message = message
    
    def __repr__(self) -> str:
        struct_str = f"[{self.structure}] " if self.structure else ""
        return f"{self.severity.upper()}: {struct_str}{self.message}"


class DatasetQualityAssessment:
    """
    Assess quality of DICOM CT and RTSTRUCT datasets for radiotherapy contouring.
    """
    
    def __init__(self, dicom_root: Path, output_dir: Path):
        """
        Initialize quality assessment.
        
        Args:
            dicom_root: Root directory containing patient DICOM folders
            output_dir: Directory to save quality reports
        """
        self.dicom_root = Path(dicom_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.case_metrics: List[Dict[str, Any]] = []
        self.case_issues: Dict[str, List[QualityIssue]] = {}
    
    def assess_case(
        self,
        case_id: str,
        ct_series_path: Path,
        rtstruct_path: Path
    ) -> Dict[str, Any]:
        """
        Assess quality of a single case.
        
        Args:
            case_id: Unique identifier for the case
            ct_series_path: Path to CT series directory
            rtstruct_path: Path to RTSTRUCT DICOM file
            
        Returns:
            Dictionary containing case metrics and issues
        """
        logger.info(f"Assessing case: {case_id}")
        
        issues: List[QualityIssue] = []
        case_data = {"case_id": case_id}
        
        try:
            # Load CT series
            ct_image = self._load_ct_series(ct_series_path)
            case_data["ct_shape"] = ct_image.GetSize()
            case_data["ct_spacing"] = ct_image.GetSpacing()
            
            # Load RTSTRUCT
            rtstruct = self._load_rtstruct(rtstruct_path)
            
            # Extract and analyze structures
            structure_metrics = self._analyze_structures(ct_image, rtstruct, issues)
            
            # Add structure metrics to case data
            for canonical_name in get_all_canonical_names():
                if canonical_name in structure_metrics:
                    metrics = structure_metrics[canonical_name]
                    case_data[f"{canonical_name}_volume_cc"] = metrics.volume_cc
                    case_data[f"{canonical_name}_num_components"] = metrics.num_components
                    case_data[f"{canonical_name}_sphericity"] = metrics.sphericity
                    case_data[f"{canonical_name}_present"] = True
                else:
                    case_data[f"{canonical_name}_volume_cc"] = np.nan
                    case_data[f"{canonical_name}_num_components"] = 0
                    case_data[f"{canonical_name}_sphericity"] = np.nan
                    case_data[f"{canonical_name}_present"] = False
                    
                    # Check if this is a critical missing structure
                    if canonical_name in ["bladder", "rectum", "femoral_head_left", "femoral_head_right"]:
                        issues.append(QualityIssue("warning", canonical_name, "Structure not found"))
            
            case_data["num_issues"] = len(issues)
            case_data["quality_score"] = self._compute_quality_score(structure_metrics, issues)
            
        except Exception as e:
            logger.error(f"Error assessing case {case_id}: {e}")
            issues.append(QualityIssue("error", None, f"Failed to assess case: {str(e)}"))
            case_data["num_issues"] = len(issues)
            case_data["quality_score"] = 0.0
        
        self.case_metrics.append(case_data)
        self.case_issues[case_id] = issues
        
        return {"metrics": case_data, "issues": issues}
    
    def _load_ct_series(self, ct_series_path: Path) -> sitk.Image:
        """Load CT series using SimpleITK."""
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(ct_series_path))
        
        if not dicom_names:
            raise ValueError(f"No DICOM files found in {ct_series_path}")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        return image
    
    def _load_rtstruct(self, rtstruct_path: Path) -> Any:
        """Load RTSTRUCT using rt-utils."""
        # Get parent directory for CT series
        ct_dir = rtstruct_path.parent
        
        try:
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(ct_dir),
                rt_struct_path=str(rtstruct_path)
            )
            return rtstruct
        except Exception as e:
            logger.error(f"Error loading RTSTRUCT: {e}")
            raise
    
    def _analyze_structures(
        self,
        ct_image: sitk.Image,
        rtstruct: Any,
        issues: List[QualityIssue]
    ) -> Dict[str, StructureMetrics]:
        """Analyze all structures in the RTSTRUCT."""
        structure_metrics = {}
        
        # Get spacing for volume calculation
        spacing = ct_image.GetSpacing()
        voxel_volume_cc = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # mm³ to cm³
        
        # Get structure names from RTSTRUCT
        roi_names = rtstruct.get_roi_names()
        
        for roi_name in roi_names:
            # Get canonical name
            canonical_name = get_canonical_name(roi_name)
            
            if canonical_name is None:
                issues.append(QualityIssue("info", roi_name, f"Unknown structure: {roi_name}"))
                continue
            
            try:
                # Get mask
                mask = rtstruct.get_roi_mask_by_name(roi_name)
                mask_array = np.array(mask, dtype=np.uint8)
                
                # Calculate metrics
                metrics = self._calculate_structure_metrics(
                    mask_array, voxel_volume_cc, canonical_name, issues
                )
                
                structure_metrics[canonical_name] = metrics
                
            except Exception as e:
                logger.warning(f"Error processing structure {roi_name}: {e}")
                issues.append(QualityIssue("warning", canonical_name, f"Error processing: {str(e)}"))
        
        return structure_metrics
    
    def _calculate_structure_metrics(
        self,
        mask: np.ndarray,
        voxel_volume_cc: float,
        structure_name: str,
        issues: List[QualityIssue]
    ) -> StructureMetrics:
        """Calculate metrics for a single structure."""
        # Volume
        num_voxels = np.sum(mask)
        volume_cc = num_voxels * voxel_volume_cc
        
        # Connected components
        labeled_mask, num_components = scipy_label(mask)
        
        # Get region properties
        props = regionprops(labeled_mask)
        
        if len(props) > 0:
            largest_region = max(props, key=lambda x: x.area)
            centroid = largest_region.centroid
            bbox = largest_region.bbox  # (min_row, min_col, min_slice, max_row, max_col, max_slice)
            
            # Sphericity approximation
            sphericity = self._calculate_sphericity(largest_region)
        else:
            centroid = (0, 0, 0)
            bbox = (0, 0, 0, 0, 0, 0)
            sphericity = 0.0
        
        # Check volume range
        volume_range = get_volume_range(structure_name)
        if volume_range:
            min_vol, max_vol, typical_min, typical_max = volume_range
            
            if volume_cc < min_vol or volume_cc > max_vol:
                issues.append(QualityIssue(
                    "error",
                    structure_name,
                    f"Volume {volume_cc:.1f} cc outside valid range [{min_vol}, {max_vol}]"
                ))
            elif volume_cc < typical_min or volume_cc > typical_max:
                issues.append(QualityIssue(
                    "warning",
                    structure_name,
                    f"Volume {volume_cc:.1f} cc outside typical range [{typical_min}, {typical_max}]"
                ))
        
        # Check for multiple disconnected components
        if num_components > 1:
            issues.append(QualityIssue(
                "warning",
                structure_name,
                f"Structure has {num_components} disconnected components"
            ))
        
        return StructureMetrics(
            name=structure_name,
            volume_cc=volume_cc,
            centroid=centroid,
            bounding_box=bbox,
            num_components=num_components,
            sphericity=sphericity
        )
    
    @staticmethod
    def _calculate_sphericity(region) -> float:
        """Calculate sphericity of a region (0-1, 1 = perfect sphere)."""
        if region.area == 0:
            return 0.0
        
        # Approximate sphericity using area and equivalent diameter
        try:
            # For 3D: sphericity = (π^(1/3) * (6*V)^(2/3)) / A
            # Simplified approximation for computational efficiency
            volume = region.area
            surface_area = region.perimeter if hasattr(region, 'perimeter') else np.sqrt(region.area)
            
            if surface_area > 0:
                sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
                return min(sphericity, 1.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_quality_score(
        self,
        structure_metrics: Dict[str, StructureMetrics],
        issues: List[QualityIssue]
    ) -> float:
        """Compute overall quality score (0-100)."""
        score = 100.0
        
        # Penalize for errors and warnings
        for issue in issues:
            if issue.severity == "error":
                score -= 10
            elif issue.severity == "warning":
                score -= 5
            elif issue.severity == "info":
                score -= 1
        
        return max(score, 0.0)
    
    def generate_report(self, output_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate quality assessment report as DataFrame.
        
        Args:
            output_file: Optional path to save CSV report
            
        Returns:
            DataFrame with quality metrics for all cases
        """
        df = pd.DataFrame(self.case_metrics)
        
        if output_file:
            output_file = Path(output_file)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved quality report to {output_file}")
        
        return df
    
    def generate_visualizations(self, output_dir: Optional[Path] = None) -> None:
        """
        Generate quality assessment visualizations.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.case_metrics)
        
        if df.empty:
            logger.warning("No data to visualize")
            return
        
        # Volume distributions
        self._plot_volume_distributions(df, output_dir)
        
        # Quality score distribution
        self._plot_quality_scores(df, output_dir)
        
        # Structure presence heatmap
        self._plot_structure_presence(df, output_dir)
    
    def _plot_volume_distributions(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot volume distributions for each structure."""
        volume_cols = [col for col in df.columns if col.endswith("_volume_cc")]
        
        if not volume_cols:
            return
        
        fig, axes = plt.subplots(
            nrows=(len(volume_cols) + 2) // 3,
            ncols=3,
            figsize=(15, 5 * ((len(volume_cols) + 2) // 3))
        )
        axes = axes.flatten() if len(volume_cols) > 1 else [axes]
        
        for idx, col in enumerate(volume_cols):
            structure_name = col.replace("_volume_cc", "")
            data = df[col].dropna()
            
            if len(data) > 0:
                axes[idx].hist(data, bins=20, edgecolor='black')
                axes[idx].set_xlabel("Volume (cc)")
                axes[idx].set_ylabel("Count")
                axes[idx].set_title(f"{structure_name.replace('_', ' ').title()} Volume Distribution")
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(volume_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "volume_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved volume distributions to {output_dir / 'volume_distributions.png'}")
    
    def _plot_quality_scores(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot quality score distribution."""
        if "quality_score" not in df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df["quality_score"], bins=20, edgecolor='black')
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Count")
        ax.set_title("Quality Score Distribution")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "quality_scores.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved quality scores to {output_dir / 'quality_scores.png'}")
    
    def _plot_structure_presence(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot structure presence heatmap."""
        presence_cols = [col for col in df.columns if col.endswith("_present")]
        
        if not presence_cols or "case_id" not in df.columns:
            return
        
        presence_data = df[["case_id"] + presence_cols].set_index("case_id")
        presence_data.columns = [col.replace("_present", "").replace("_", " ").title() for col in presence_data.columns]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.3)))
        sns.heatmap(
            presence_data,
            cmap="RdYlGn",
            cbar_kws={"label": "Present"},
            ax=ax,
            linewidths=0.5
        )
        ax.set_title("Structure Presence Across Cases")
        ax.set_xlabel("Structure")
        ax.set_ylabel("Case")
        
        plt.tight_layout()
        plt.savefig(output_dir / "structure_presence.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved structure presence to {output_dir / 'structure_presence.png'}")
    
    def select_training_cohort(
        self,
        min_quality_score: float = 70.0,
        required_structures: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select cases suitable for training based on quality criteria.
        
        Args:
            min_quality_score: Minimum quality score threshold
            required_structures: List of structures that must be present
            
        Returns:
            List of case IDs meeting the criteria
        """
        if required_structures is None:
            required_structures = ["bladder", "rectum"]
        
        df = pd.DataFrame(self.case_metrics)
        
        # Filter by quality score
        selected = df[df["quality_score"] >= min_quality_score]
        
        # Filter by required structures
        for structure in required_structures:
            col_name = f"{structure}_present"
            if col_name in selected.columns:
                selected = selected[selected[col_name] == True]
        
        selected_ids = selected["case_id"].tolist()
        
        logger.info(f"Selected {len(selected_ids)} / {len(df)} cases for training")
        
        return selected_ids
