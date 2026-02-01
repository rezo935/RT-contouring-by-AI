"""CLI script for training nnU-Net models."""

import argparse
import logging
from pathlib import Path
import sys

from config.paths import get_path_config
from src.training.train_nnunet import NnUNetTrainer


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="Train nnU-Net models for RT auto-contouring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run nnU-Net preprocessing (planning and preprocessing)"
    )
    preprocess_parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Dataset ID"
    )
    preprocess_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset integrity before preprocessing"
    )
    preprocess_parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes for preprocessing"
    )
    preprocess_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train nnU-Net model"
    )
    train_parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Dataset ID"
    )
    train_parser.add_argument(
        "-c", "--configuration",
        type=str,
        default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="Configuration name"
    )
    train_parser.add_argument(
        "-f", "--fold",
        type=int,
        default=0,
        help="Fold number (0-4)"
    )
    train_parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Train all folds sequentially"
    )
    train_parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name"
    )
    train_parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetPlans",
        help="Plans identifier"
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (default: use nnU-Net default)"
    )
    train_parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Continue training from checkpoint"
    )
    train_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Find best configuration command
    find_best_parser = subparsers.add_parser(
        "find-best",
        help="Find best configuration based on validation performance"
    )
    find_best_parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Dataset ID"
    )
    find_best_parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetPlans",
        help="Plans identifier"
    )
    find_best_parser.add_argument(
        "--configurations",
        type=str,
        nargs="+",
        default=None,
        help="Configurations to compare (default: all available)"
    )
    find_best_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check training status"
    )
    status_parser.add_argument(
        "-d", "--dataset-id",
        type=int,
        required=True,
        help="Dataset ID"
    )
    status_parser.add_argument(
        "-c", "--configuration",
        type=str,
        default="3d_fullres",
        help="Configuration name"
    )
    status_parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name"
    )
    status_parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetPlans",
        help="Plans identifier"
    )
    status_parser.add_argument(
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
    
    # Get path config and setup nnU-Net environment
    path_config = get_path_config()
    path_config.setup_nnunet_env()
    
    # Initialize trainer
    trainer = NnUNetTrainer(path_config)
    
    if args.command == "preprocess":
        run_preprocess(args, trainer, logger)
    elif args.command == "train":
        run_train(args, trainer, logger)
    elif args.command == "find-best":
        run_find_best(args, trainer, logger)
    elif args.command == "status":
        run_status(args, trainer, logger)


def run_preprocess(args, trainer, logger):
    """Run preprocessing."""
    logger.info(f"Running nnU-Net preprocessing for dataset {args.dataset_id}")
    
    success = trainer.plan_and_preprocess(
        dataset_id=args.dataset_id,
        verify_dataset_integrity=args.verify,
        num_processes=args.num_processes
    )
    
    if success:
        logger.info("\nPreprocessing completed successfully!")
    else:
        logger.error("\nPreprocessing failed!")
        sys.exit(1)


def run_train(args, trainer, logger):
    """Run training."""
    logger.info(f"Starting nnU-Net training")
    logger.info(f"  Dataset: {args.dataset_id}")
    logger.info(f"  Configuration: {args.configuration}")
    
    if args.all_folds:
        logger.info(f"  Training all folds (0-4)")
        success = trainer.train_all_folds(
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            trainer=args.trainer,
            plans=args.plans,
            num_epochs=args.num_epochs
        )
    else:
        logger.info(f"  Fold: {args.fold}")
        success = trainer.train(
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            fold=args.fold,
            trainer=args.trainer,
            plans=args.plans,
            num_epochs=args.num_epochs,
            continue_training=args.continue_training
        )
    
    if success:
        logger.info("\nTraining completed successfully!")
    else:
        logger.error("\nTraining failed!")
        sys.exit(1)


def run_find_best(args, trainer, logger):
    """Find best configuration."""
    logger.info(f"Finding best configuration for dataset {args.dataset_id}")
    
    best_config = trainer.find_best_configuration(
        dataset_id=args.dataset_id,
        plans=args.plans,
        configurations=args.configurations
    )
    
    logger.info("\nCheck the output above for the best configuration.")


def run_status(args, trainer, logger):
    """Check training status."""
    logger.info(f"Checking training status")
    logger.info(f"  Dataset: {args.dataset_id}")
    logger.info(f"  Configuration: {args.configuration}")
    
    status = trainer.get_training_status(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        trainer=args.trainer,
        plans=args.plans
    )
    
    logger.info(f"\nTraining Status:")
    logger.info(f"  Results directory: {status['results_dir']}")
    logger.info(f"  Exists: {status['exists']}")
    
    if status['exists']:
        logger.info(f"  Folds trained: {len(status['folds_trained'])}")
        
        for fold_status in status['folds_trained']:
            logger.info(f"\n  Fold {fold_status['fold']}:")
            logger.info(f"    Directory exists: {fold_status['directory_exists']}")
            logger.info(f"    Final checkpoint: {fold_status['checkpoint_final']}")
            logger.info(f"    Best checkpoint: {fold_status['checkpoint_best']}")


if __name__ == "__main__":
    main()
