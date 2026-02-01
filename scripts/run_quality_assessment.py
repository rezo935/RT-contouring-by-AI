"""CLI script for running quality assessment on DICOM datasets."""

import argparse
import logging
from pathlib import Path
import sys

from config.paths import get_path_config
from src.data_quality.quality_assessment import DatasetQualityAssessment
from src.preprocessing.dicom_to_nifti import find_dicom_files


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for quality assessment CLI."""
    parser = argparse.ArgumentParser(
        description="Run quality assessment on RT contouring DICOM datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "dicom_root",
        type=str,
        help="Root directory containing patient DICOM folders"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for quality reports (default: data/quality_reports)"
    )
    
    parser.add_argument(
        "-r", "--report-name",
        type=str,
        default="quality_assessment.csv",
        help="Name of the output CSV report"
    )
    
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=70.0,
        help="Minimum quality score for training cohort selection"
    )
    
    parser.add_argument(
        "--required-structures",
        type=str,
        nargs="+",
        default=["bladder", "rectum"],
        help="Structures that must be present for training cohort"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--select-cohort",
        action="store_true",
        help="Select training cohort based on quality criteria"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Get path config
    path_config = get_path_config()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = path_config.quality_reports
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting quality assessment")
    logger.info(f"  DICOM root: {args.dicom_root}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Find patient directories
    dicom_root = Path(args.dicom_root)
    
    if not dicom_root.exists():
        logger.error(f"DICOM root directory does not exist: {dicom_root}")
        sys.exit(1)
    
    # Get list of patient directories
    patient_dirs = [d for d in dicom_root.iterdir() if d.is_dir()]
    logger.info(f"Found {len(patient_dirs)} patient directories")
    
    # Initialize quality assessment
    qa = DatasetQualityAssessment(
        dicom_root=dicom_root,
        output_dir=output_dir
    )
    
    # Process each patient
    successful = 0
    failed = 0
    
    for patient_dir in patient_dirs:
        case_id = patient_dir.name
        
        try:
            # Find CT and RTSTRUCT files
            ct_series_path, rtstruct_path = find_dicom_files(patient_dir)
            
            if ct_series_path is None:
                logger.warning(f"No CT series found for {case_id}")
                failed += 1
                continue
            
            if rtstruct_path is None:
                logger.warning(f"No RTSTRUCT found for {case_id}")
                failed += 1
                continue
            
            # Assess case
            qa.assess_case(
                case_id=case_id,
                ct_series_path=ct_series_path,
                rtstruct_path=rtstruct_path
            )
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Error assessing {case_id}: {e}")
            failed += 1
    
    logger.info(f"\nQuality assessment completed:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    
    # Generate report
    report_path = output_dir / args.report_name
    df = qa.generate_report(report_path)
    
    logger.info(f"\nQuality Report Summary:")
    logger.info(f"  Total cases: {len(df)}")
    if len(df) > 0:
        logger.info(f"  Mean quality score: {df['quality_score'].mean():.1f}")
        logger.info(f"  Cases with score >= {args.min_quality_score}: {len(df[df['quality_score'] >= args.min_quality_score])}")
    
    # Generate visualizations
    if args.visualize:
        logger.info("\nGenerating visualizations...")
        qa.generate_visualizations(output_dir)
    
    # Select training cohort
    if args.select_cohort:
        logger.info("\nSelecting training cohort...")
        selected_cases = qa.select_training_cohort(
            min_quality_score=args.min_quality_score,
            required_structures=args.required_structures
        )
        
        # Save selected cases list
        cohort_file = output_dir / "training_cohort.txt"
        with open(cohort_file, 'w') as f:
            for case_id in selected_cases:
                f.write(f"{case_id}\n")
        
        logger.info(f"Saved training cohort to {cohort_file}")
        logger.info(f"Selected {len(selected_cases)} cases for training")
    
    logger.info("\nQuality assessment complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
