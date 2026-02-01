"""nnU-Net training wrapper.

This module provides wrapper functions for nnU-Net training operations,
including environment setup, preprocessing, training, and finding best configurations.
"""

from pathlib import Path
from typing import Optional, List
import os
import subprocess
import logging

from config.paths import PathConfig

logger = logging.getLogger(__name__)


class NnUNetTrainer:
    """
    Wrapper for nnU-Net training operations.
    
    This class provides convenient methods for running nnU-Net preprocessing,
    training, and evaluation while handling environment setup.
    """
    
    def __init__(self, path_config: PathConfig):
        """
        Initialize nnU-Net trainer.
        
        Args:
            path_config: Path configuration object
        """
        self.path_config = path_config
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set up nnU-Net environment variables."""
        self.path_config.setup_nnunet_env()
        
        logger.info(f"nnU-Net environment configured:")
        logger.info(f"  nnUNet_raw: {os.environ['nnUNet_raw']}")
        logger.info(f"  nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
        logger.info(f"  nnUNet_results: {os.environ['nnUNet_results']}")
    
    def plan_and_preprocess(
        self,
        dataset_id: int,
        verify_dataset_integrity: bool = True,
        num_processes: int = 8
    ) -> bool:
        """
        Run nnU-Net preprocessing (planning and preprocessing).
        
        Args:
            dataset_id: Dataset ID (e.g., 1 for Dataset001)
            verify_dataset_integrity: Whether to verify dataset first
            num_processes: Number of processes for preprocessing
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running nnU-Net preprocessing for dataset {dataset_id}")
        
        try:
            # Verify dataset integrity first
            if verify_dataset_integrity:
                cmd_verify = [
                    "nnUNetv2_plan_and_preprocess",
                    "-d", str(dataset_id),
                    "--verify_dataset_integrity"
                ]
                
                logger.info(f"Verifying dataset integrity: {' '.join(cmd_verify)}")
                result = subprocess.run(
                    cmd_verify,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.error(f"Dataset verification failed: {result.stderr}")
                    return False
                
                logger.info("Dataset verification passed")
            
            # Run preprocessing
            cmd_preprocess = [
                "nnUNetv2_plan_and_preprocess",
                "-d", str(dataset_id),
                "-np", str(num_processes)
            ]
            
            logger.info(f"Running preprocessing: {' '.join(cmd_preprocess)}")
            result = subprocess.run(
                cmd_preprocess,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Preprocessing failed: {result.stderr}")
                return False
            
            logger.info("Preprocessing completed successfully")
            logger.debug(f"Output: {result.stdout}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return False
    
    def train(
        self,
        dataset_id: int,
        configuration: str = "3d_fullres",
        fold: int = 0,
        trainer: str = "nnUNetTrainer",
        plans: str = "nnUNetPlans",
        num_epochs: Optional[int] = None,
        continue_training: bool = False,
        validation_only: bool = False
    ) -> bool:
        """
        Train nnU-Net model.
        
        Args:
            dataset_id: Dataset ID
            configuration: Configuration name (e.g., "3d_fullres", "2d", "3d_lowres")
            fold: Fold number (0-4 for 5-fold CV, or "all" for all folds)
            trainer: Trainer class name
            plans: Plans identifier
            num_epochs: Number of epochs (None for default)
            continue_training: Continue from checkpoint
            validation_only: Only run validation
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Training nnU-Net model:")
        logger.info(f"  Dataset: {dataset_id}")
        logger.info(f"  Configuration: {configuration}")
        logger.info(f"  Fold: {fold}")
        logger.info(f"  Trainer: {trainer}")
        
        try:
            cmd = [
                "nnUNetv2_train",
                str(dataset_id),
                configuration,
                str(fold),
                "-tr", trainer,
                "-p", plans
            ]
            
            if num_epochs is not None:
                cmd.extend(["--npz", str(num_epochs)])
            
            if continue_training:
                cmd.append("-c")
            
            if validation_only:
                cmd.append("--val")
            
            logger.info(f"Running training: {' '.join(cmd)}")
            
            # Run training (this will take a long time)
            result = subprocess.run(
                cmd,
                capture_output=False,  # Stream output to console
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error("Training failed")
                return False
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def train_all_folds(
        self,
        dataset_id: int,
        configuration: str = "3d_fullres",
        trainer: str = "nnUNetTrainer",
        plans: str = "nnUNetPlans",
        num_epochs: Optional[int] = None
    ) -> bool:
        """
        Train all folds sequentially.
        
        Args:
            dataset_id: Dataset ID
            configuration: Configuration name
            trainer: Trainer class name
            plans: Plans identifier
            num_epochs: Number of epochs per fold
            
        Returns:
            True if all folds successful, False otherwise
        """
        logger.info("Training all folds (0-4)")
        
        for fold in range(5):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training fold {fold}")
            logger.info(f"{'='*60}\n")
            
            success = self.train(
                dataset_id=dataset_id,
                configuration=configuration,
                fold=fold,
                trainer=trainer,
                plans=plans,
                num_epochs=num_epochs
            )
            
            if not success:
                logger.error(f"Training failed for fold {fold}")
                return False
        
        logger.info("All folds trained successfully")
        return True
    
    def find_best_configuration(
        self,
        dataset_id: int,
        plans: str = "nnUNetPlans",
        configurations: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Find the best configuration based on validation performance.
        
        Args:
            dataset_id: Dataset ID
            plans: Plans identifier
            configurations: List of configurations to compare (None for default)
            
        Returns:
            Best configuration name, or None if failed
        """
        logger.info(f"Finding best configuration for dataset {dataset_id}")
        
        try:
            cmd = [
                "nnUNetv2_find_best_configuration",
                str(dataset_id),
                "-p", plans
            ]
            
            if configurations:
                cmd.extend(["-c"] + configurations)
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to find best configuration: {result.stderr}")
                return None
            
            # Parse output to find best configuration
            output = result.stdout
            logger.info(f"Find best configuration output:\n{output}")
            
            # The output typically contains a line like "Best configuration: 3d_fullres"
            for line in output.split('\n'):
                if "best" in line.lower() and "configuration" in line.lower():
                    logger.info(f"Found: {line}")
            
            return None  # User should check output logs
            
        except Exception as e:
            logger.error(f"Error finding best configuration: {e}")
            return None
    
    def predict(
        self,
        input_folder: Path,
        output_folder: Path,
        dataset_id: int,
        configuration: str = "3d_fullres",
        trainer: str = "nnUNetTrainer",
        plans: str = "nnUNetPlans",
        folds: str = "all",
        save_probabilities: bool = False
    ) -> bool:
        """
        Run inference using trained model.
        
        Args:
            input_folder: Folder with input images
            output_folder: Folder for output predictions
            dataset_id: Dataset ID
            configuration: Configuration name
            trainer: Trainer class name
            plans: Plans identifier
            folds: Which folds to use ("all" or specific fold numbers)
            save_probabilities: Whether to save probability maps
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running inference:")
        logger.info(f"  Input: {input_folder}")
        logger.info(f"  Output: {output_folder}")
        logger.info(f"  Model: Dataset{dataset_id:03d}_{configuration}")
        
        try:
            cmd = [
                "nnUNetv2_predict",
                "-i", str(input_folder),
                "-o", str(output_folder),
                "-d", str(dataset_id),
                "-c", configuration,
                "-tr", trainer,
                "-p", plans,
                "-f", str(folds)
            ]
            
            if save_probabilities:
                cmd.append("--save_probabilities")
            
            logger.info(f"Running prediction: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error("Prediction failed")
                return False
            
            logger.info("Prediction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return False
    
    def get_training_status(
        self,
        dataset_id: int,
        configuration: str = "3d_fullres",
        trainer: str = "nnUNetTrainer",
        plans: str = "nnUNetPlans"
    ) -> dict:
        """
        Get training status for a dataset.
        
        Args:
            dataset_id: Dataset ID
            configuration: Configuration name
            trainer: Trainer class name
            plans: Plans identifier
            
        Returns:
            Dictionary with training status information
        """
        results_dir = self.path_config.nnunet_results / f"Dataset{dataset_id:03d}_{trainer}__{plans}__{configuration}"
        
        status = {
            "dataset_id": dataset_id,
            "configuration": configuration,
            "results_dir": str(results_dir),
            "exists": results_dir.exists(),
            "folds_trained": []
        }
        
        if results_dir.exists():
            # Check which folds have been trained
            for fold in range(5):
                fold_dir = results_dir / f"fold_{fold}"
                if fold_dir.exists():
                    checkpoint_final = fold_dir / "checkpoint_final.pth"
                    checkpoint_best = fold_dir / "checkpoint_best.pth"
                    
                    fold_status = {
                        "fold": fold,
                        "directory_exists": True,
                        "checkpoint_final": checkpoint_final.exists(),
                        "checkpoint_best": checkpoint_best.exists()
                    }
                    
                    status["folds_trained"].append(fold_status)
        
        return status
