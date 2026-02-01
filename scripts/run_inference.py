"""CLI script for running inference on new cases."""

import argparse
import logging
from pathlib import Path
import sys

from config.paths import get_path_config
from src.inference.predict import InferenceEngine
from src.inference.nifti_to_rtstruct import NiftiToRtstructConverter
from src.preprocessing.dicom_to_nifti import find_dicom_files


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for inference CLI."""
    parser = argparse.ArgumentParser(
        description="Run inference on new RT cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Input path (DICOM directory or NIfTI file)"
    )
    
    parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Trained dataset ID"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/inference_output)"
    )
    
    parser.add_argument(
        "-c", "--configuration",
        type=str,
        default="3d_fullres",
        help="Model configuration"
    )
    
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name"
    )
    
    parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetPlans",
        help="Plans identifier"
    )
    
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Which folds to use for ensemble (default: all)"
    )
    
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Case identifier (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--create-rtstruct",
        action="store_true",
        help="Create DICOM RTSTRUCT output"
    )
    
    parser.add_argument(
        "--input-type",
        type=str,
        choices=["dicom", "nifti", "auto"],
        default="auto",
        help="Input type (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple cases (input_path should be a directory)"
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
    path_config.setup_nnunet_env()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = path_config.inference_output
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running inference")
    logger.info(f"  Input: {args.input_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Dataset: {args.dataset_id}")
    logger.info(f"  Configuration: {args.configuration}")
    
    # Initialize inference engine
    try:
        engine = InferenceEngine(
            dataset_id=args.dataset_id,
            path_config=path_config,
            configuration=args.configuration,
            trainer=args.trainer,
            plans=args.plans,
            folds=args.folds
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        sys.exit(1)
    
    # Initialize RTSTRUCT converter if needed
    rtstruct_converter = None
    if args.create_rtstruct:
        rtstruct_converter = NiftiToRtstructConverter()
    
    # Process input
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    if args.batch:
        run_batch_inference(
            engine=engine,
            input_dir=input_path,
            output_dir=output_dir,
            rtstruct_converter=rtstruct_converter,
            input_type=args.input_type,
            logger=logger
        )
    else:
        run_single_inference(
            engine=engine,
            input_path=input_path,
            output_dir=output_dir,
            case_id=args.case_id,
            rtstruct_converter=rtstruct_converter,
            input_type=args.input_type,
            logger=logger
        )


def detect_input_type(input_path: Path) -> str:
    """Detect whether input is DICOM or NIfTI."""
    if input_path.is_file():
        if input_path.suffix in ['.nii', '.gz']:
            return "nifti"
        elif input_path.suffix == '.dcm':
            return "dicom"
    elif input_path.is_dir():
        # Check for DICOM files
        dcm_files = list(input_path.glob("*.dcm"))
        if dcm_files:
            return "dicom"
        # Check for NIfTI files
        nii_files = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
        if nii_files:
            return "nifti"
    
    return "unknown"


def run_single_inference(
    engine,
    input_path: Path,
    output_dir: Path,
    case_id: str,
    rtstruct_converter,
    input_type: str,
    logger
):
    """Run inference on a single case."""
    # Detect input type if auto
    if input_type == "auto":
        input_type = detect_input_type(input_path)
        logger.info(f"Detected input type: {input_type}")
    
    if input_type == "unknown":
        logger.error("Could not determine input type")
        sys.exit(1)
    
    # Generate case ID if not provided
    if case_id is None:
        case_id = f"case_{input_path.stem}"
    
    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run inference
        if input_type == "dicom":
            # Find CT series
            ct_series_path, _ = find_dicom_files(input_path)
            
            if ct_series_path is None:
                logger.error(f"No CT series found in {input_path}")
                sys.exit(1)
            
            result = engine.predict_from_dicom(
                ct_series_path=ct_series_path,
                output_dir=case_output_dir,
                case_id=case_id
            )
        else:  # nifti
            result = engine.predict_from_nifti(
                ct_nifti_path=input_path,
                output_dir=case_output_dir,
                case_id=case_id
            )
        
        logger.info(f"\nInference completed successfully!")
        logger.info(f"  Segmentation: {result['segmentation']}")
        
        # Create RTSTRUCT if requested
        if rtstruct_converter and input_type == "dicom":
            logger.info(f"\nCreating DICOM RTSTRUCT...")
            
            ct_series_path, _ = find_dicom_files(input_path)
            rtstruct_path = case_output_dir / f"{case_id}_RTSTRUCT.dcm"
            
            rtstruct_converter.convert(
                segmentation_path=result['segmentation'],
                ct_series_path=ct_series_path,
                output_path=rtstruct_path
            )
            
            logger.info(f"  RTSTRUCT: {rtstruct_path}")
        
        logger.info(f"\nResults saved to: {case_output_dir}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


def run_batch_inference(
    engine,
    input_dir: Path,
    output_dir: Path,
    rtstruct_converter,
    input_type: str,
    logger
):
    """Run inference on multiple cases."""
    logger.info(f"Running batch inference on {input_dir}")
    
    # Detect input type if auto
    if input_type == "auto":
        input_type = detect_input_type(input_dir)
        logger.info(f"Detected input type: {input_type}")
    
    # Get list of cases
    if input_type == "dicom":
        # Each subdirectory is a case
        case_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        input_cases = []
        for case_dir in case_dirs:
            ct_series_path, _ = find_dicom_files(case_dir)
            if ct_series_path:
                input_cases.append({
                    "case_id": case_dir.name,
                    "ct_series_path": ct_series_path
                })
    else:  # nifti
        # Each NIfTI file is a case
        nifti_files = list(input_dir.glob("*_0000.nii.gz"))
        
        input_cases = []
        for nifti_file in nifti_files:
            case_id = nifti_file.name.replace("_0000.nii.gz", "")
            input_cases.append({
                "case_id": case_id,
                "ct_nifti_path": nifti_file
            })
    
    logger.info(f"Found {len(input_cases)} cases to process")
    
    # Run batch inference
    results = engine.batch_predict(
        input_cases=input_cases,
        output_dir=output_dir
    )
    
    # Create RTSTRUCTs if requested
    if rtstruct_converter and input_type == "dicom":
        logger.info(f"\nCreating DICOM RTSTRUCTs...")
        
        rtstruct_cases = []
        for result in results:
            if "error" not in result:
                case_id = result["case_id"]
                # Find original CT series
                case_dir = input_dir / case_id
                ct_series_path, _ = find_dicom_files(case_dir)
                
                if ct_series_path:
                    rtstruct_cases.append({
                        "case_id": case_id,
                        "segmentation_path": result["segmentation"],
                        "ct_series_path": ct_series_path
                    })
        
        rtstruct_converter.convert_batch(
            cases=rtstruct_cases,
            output_dir=output_dir
        )
    
    # Summary
    successful = len([r for r in results if "error" not in r])
    failed = len([r for r in results if "error" in r])
    
    logger.info(f"\nBatch inference completed:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
