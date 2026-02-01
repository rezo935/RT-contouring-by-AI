"""CLI script for preprocessing DICOM data and preparing nnU-Net datasets."""

import argparse
import logging
from pathlib import Path
import sys

from config.paths import get_path_config
from src.preprocessing.dicom_to_nifti import DicomToNiftiConverter, find_dicom_files
from src.preprocessing.prepare_dataset import NnUNetDatasetPreparator


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="Preprocess DICOM data and prepare nnU-Net datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Convert DICOM to NIfTI
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert DICOM CT and RTSTRUCT to NIfTI format"
    )
    convert_parser.add_argument(
        "dicom_root",
        type=str,
        help="Root directory containing patient DICOM folders"
    )
    convert_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for NIfTI files"
    )
    convert_parser.add_argument(
        "--cohort-file",
        type=str,
        default=None,
        help="Text file with list of case IDs to process (one per line)"
    )
    convert_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Prepare nnU-Net dataset
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare nnU-Net dataset from NIfTI files"
    )
    prepare_parser.add_argument(
        "nifti_dir",
        type=str,
        help="Directory containing NIfTI files"
    )
    prepare_parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Dataset ID (e.g., 1 for Dataset001)"
    )
    prepare_parser.add_argument(
        "-n", "--dataset-name",
        type=str,
        default="PelvisOAR",
        help="Dataset name"
    )
    prepare_parser.add_argument(
        "--description",
        type=str,
        default="Pelvis OAR Segmentation",
        help="Dataset description"
    )
    prepare_parser.add_argument(
        "--modality",
        type=str,
        default="CT",
        help="Imaging modality"
    )
    prepare_parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.8,
        help="Train/validation split ratio"
    )
    prepare_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Full pipeline (convert + prepare)
    full_parser = subparsers.add_parser(
        "full",
        help="Run full pipeline: convert DICOM to NIfTI and prepare nnU-Net dataset"
    )
    full_parser.add_argument(
        "dicom_root",
        type=str,
        help="Root directory containing patient DICOM folders"
    )
    full_parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Dataset ID"
    )
    full_parser.add_argument(
        "-n", "--dataset-name",
        type=str,
        default="PelvisOAR",
        help="Dataset name"
    )
    full_parser.add_argument(
        "--cohort-file",
        type=str,
        default=None,
        help="Text file with list of case IDs to process"
    )
    full_parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.8,
        help="Train/validation split ratio"
    )
    full_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Get path config
    path_config = get_path_config()
    path_config.create_directories()
    
    if args.command == "convert":
        run_convert(args, path_config, logger)
    elif args.command == "prepare":
        run_prepare(args, path_config, logger)
    elif args.command == "full":
        run_full_pipeline(args, path_config, logger)


def run_convert(args, path_config, logger):
    """Run DICOM to NIfTI conversion."""
    dicom_root = Path(args.dicom_root)
    
    if not dicom_root.exists():
        logger.error(f"DICOM root directory does not exist: {dicom_root}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = path_config.dicom_processed / "nifti"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting DICOM to NIfTI")
    logger.info(f"  Input: {dicom_root}")
    logger.info(f"  Output: {output_dir}")
    
    # Load cohort file if provided
    case_ids = None
    if args.cohort_file:
        cohort_file = Path(args.cohort_file)
        if cohort_file.exists():
            with open(cohort_file, 'r') as f:
                case_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(case_ids)} case IDs from cohort file")
    
    # Get patient directories
    patient_dirs = [d for d in dicom_root.iterdir() if d.is_dir()]
    
    if case_ids:
        patient_dirs = [d for d in patient_dirs if d.name in case_ids]
    
    logger.info(f"Processing {len(patient_dirs)} cases")
    
    # Initialize converter
    converter = DicomToNiftiConverter()
    
    # Process each case
    successful = 0
    failed = 0
    
    for patient_dir in patient_dirs:
        case_id = patient_dir.name
        
        try:
            # Find CT and RTSTRUCT
            ct_series_path, rtstruct_path = find_dicom_files(patient_dir)
            
            if ct_series_path is None or rtstruct_path is None:
                logger.warning(f"Missing CT or RTSTRUCT for {case_id}")
                failed += 1
                continue
            
            # Convert
            converter.convert_case(
                case_id=case_id,
                ct_series_path=ct_series_path,
                rtstruct_path=rtstruct_path,
                output_dir=output_dir
            )
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Error converting {case_id}: {e}")
            failed += 1
    
    logger.info(f"\nConversion completed:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")


def run_prepare(args, path_config, logger):
    """Prepare nnU-Net dataset."""
    nifti_dir = Path(args.nifti_dir)
    
    if not nifti_dir.exists():
        logger.error(f"NIfTI directory does not exist: {nifti_dir}")
        sys.exit(1)
    
    logger.info(f"Preparing nnU-Net dataset")
    logger.info(f"  Dataset ID: {args.dataset_id}")
    logger.info(f"  Dataset name: {args.dataset_name}")
    logger.info(f"  NIfTI directory: {nifti_dir}")
    
    # Initialize preparator
    preparator = NnUNetDatasetPreparator(path_config)
    
    # Setup nnU-Net environment
    path_config.setup_nnunet_env()
    
    # Create dataset
    dataset_path = preparator.create_dataset(
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        nifti_source_dir=nifti_dir,
        description=args.description,
        modality=args.modality,
        train_val_split=args.train_val_split
    )
    
    # Verify dataset
    if preparator.verify_dataset(args.dataset_id, args.dataset_name):
        logger.info("\nDataset verification passed!")
    else:
        logger.error("\nDataset verification failed!")
        sys.exit(1)
    
    # Print statistics
    stats = preparator.get_dataset_statistics(args.dataset_id, args.dataset_name)
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Path: {stats['dataset_path']}")
    logger.info(f"  Images: {stats.get('num_images', 0)}")
    logger.info(f"  Labels: {stats.get('num_labels', 0)}")
    logger.info(f"  Folds: {stats.get('num_folds', 0)}")
    
    logger.info(f"\nDataset ready for nnU-Net training!")


def run_full_pipeline(args, path_config, logger):
    """Run full preprocessing pipeline."""
    logger.info("Running full preprocessing pipeline")
    
    # Step 1: Convert DICOM to NIfTI
    temp_nifti_dir = path_config.dicom_processed / "nifti_temp"
    
    class ConvertArgs:
        dicom_root = args.dicom_root
        output_dir = str(temp_nifti_dir)
        cohort_file = args.cohort_file
        verbose = args.verbose
    
    run_convert(ConvertArgs(), path_config, logger)
    
    # Step 2: Prepare nnU-Net dataset
    class PrepareArgs:
        nifti_dir = str(temp_nifti_dir)
        dataset_id = args.dataset_id
        dataset_name = args.dataset_name
        description = "Pelvis OAR Segmentation"
        modality = "CT"
        train_val_split = args.train_val_split
        verbose = args.verbose
    
    run_prepare(PrepareArgs(), path_config, logger)
    
    logger.info("\nFull pipeline completed successfully!")


if __name__ == "__main__":
    main()
