#!/usr/bin/env python
"""
Main training script for the car price prediction project.

This script orchestrates the entire training pipeline including:
- Data loading and validation
- Preprocessing and feature engineering
- Model training and hyperparameter tuning
- Evaluation and model selection
- Saving the best model
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.config.config_loader import get_config
from src.data.data_loader import DataLoader, load_sample_data
from src.data.data_preprocessing import DataPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.utils.helpers import setup_logging, create_directories, set_random_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train car price prediction models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use sample data for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_path=project_root / "logs" / "training.log"
    )
    
    logger.info("="*60)
    logger.info("Car Price Prediction - Training Pipeline")
    logger.info("="*60)
    
    # Load configuration
    config = get_config(args.config)
    
    # Set random seed for reproducibility
    set_random_seed(config.get('data.random_state', 42))
    
    # Create necessary directories
    create_directories([
        project_root / "models",
        project_root / "logs",
        project_root / "reports" / "figures",
        project_root / "data" / "processed"
    ])
    
    try:
        # Step 1: Load Data
        logger.info("\n" + "="*40)
        logger.info("Step 1: Loading Data")
        logger.info("="*40)
        
        data_loader = DataLoader()
        
        if args.sample_data:
            logger.info("Using sample data for testing")
            df = load_sample_data()
        else:
            data_path = args.data_path or config.get('data.raw_path')
            df = data_loader.load_csv(project_root / data_path)
        
        # Validate data
        is_valid, issues = data_loader.validate_data(df)
        if not is_valid:
            logger.error(f"Data validation failed: {issues}")
            # Continue with warnings
        
        # Get data summary
        summary = data_loader.get_data_summary(df)
        logger.info(f"Data shape: {summary['shape']}")
        logger.info(f"Memory usage: {summary['memory_usage']:.2f} MB")
        
        # Step 2: Preprocess Data
        logger.info("\n" + "="*40)
        logger.info("Step 2: Preprocessing Data")
        logger.info("="*40)
        
        preprocessor = DataPreprocessor()
        
        # Clean data
        df_clean = preprocessor.clean_data(df)
        logger.info(f"After cleaning: {df_clean.shape}")
        
        # Create features
        df_features = preprocessor.create_features(df_clean)
        logger.info(f"After feature creation: {df_features.shape}")
        
        # Step 3: Split Data
        logger.info("\n" + "="*40)
        logger.info("Step 3: Splitting Data")
        logger.info("="*40)
        
        test_size = config.get('data.test_size', 0.2)
        train_df, test_df = data_loader.split_data(
            df_features,
            test_size=test_size,
            random_state=config.get('data.random_state', 42),
            save=True
        )
        
        # Step 4: Prepare Features
        logger.info("\n" + "="*40)
        logger.info("Step 4: Preparing Features")
        logger.info("="*40)
        
        # Prepare training features
        X_train, y_train = preprocessor.prepare_features(train_df, fit=True)
        logger.info(f"Training features shape: {X_train.shape}")
        
        # Prepare test features (using fitted preprocessor)
        X_test, y_test = preprocessor.prepare_features(test_df, fit=False)
        logger.info(f"Test features shape: {X_test.shape}")
        
        # Save preprocessor
        preprocessor.save_preprocessor(project_root / args.output_dir / "preprocessor.pkl")
        
        # Get feature names
        feature_names = preprocessor.get_feature_names()
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Step 5: Feature Analysis
        logger.info("\n" + "="*40)
        logger.info("Step 5: Feature Analysis")
        logger.info("="*40)
        
        feature_builder = FeatureBuilder()
        
        # Analyze correlations
        correlations = feature_builder.analyze_correlations(df_features)
        if correlations['target_correlations']:
            logger.info("Top 5 features correlated with target:")
            for feature, corr in correlations['target_correlations'][:5]:
                logger.info(f"  {feature}: {corr:.4f}")
        
        # Step 6: Train Models
        logger.info("\n" + "="*40)
        logger.info("Step 6: Training Models")
        logger.info("="*40)
        
        trainer = ModelTrainer()
        
        # Train all configured models
        trainer.train_all_models(
            X_train, y_train,
            tune_hyperparameters=args.tune
        )
        
        # Get training summary
        training_summary = trainer.get_training_summary()
        logger.info("\nTraining Summary:")
        logger.info(training_summary.to_string())
        
        # Step 7: Evaluate Models
        logger.info("\n" + "="*40)
        logger.info("Step 7: Evaluating Models")
        logger.info("="*40)
        
        evaluator = ModelEvaluator()
        
        # Evaluate all models
        comparison_df = evaluator.evaluate_all_models(
            trainer.models,
            X_test,
            y_test
        )
        
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string())
        
        # Step 8: Select Best Model
        logger.info("\n" + "="*40)
        logger.info("Step 8: Selecting Best Model")
        logger.info("="*40)
        
        best_name, best_model = trainer.select_best_model(X_test, y_test)
        logger.info(f"Best model: {best_name}")
        
        # Step 9: Save Models
        logger.info("\n" + "="*40)
        logger.info("Step 9: Saving Models")
        logger.info("="*40)
        
        trainer.save_models(
            save_all=config.get('model_saving.save_all_models', True),
            save_path=project_root / args.output_dir
        )
        
        # Step 10: Generate Reports
        logger.info("\n" + "="*40)
        logger.info("Step 10: Generating Reports")
        logger.info("="*40)
        
        # Generate plots
        evaluator.plot_predictions(
            save_path=project_root / "reports" / "figures" / "predictions.png"
        )
        
        evaluator.plot_model_comparison(
            save_path=project_root / "reports" / "figures" / "model_comparison.png"
        )
        
        evaluator.plot_residuals(
            best_name,
            save_path=project_root / "reports" / "figures" / f"residuals_{best_name}.png"
        )
        
        # Feature importance
        importance_df = trainer.get_feature_importance(
            model_name=best_name,
            feature_names=feature_names
        )
        if not importance_df.empty:
            evaluator.plot_feature_importance(
                importance_df,
                save_path=project_root / "reports" / "figures" / "feature_importance.png"
            )
        
        # Generate report
        report = evaluator.generate_report(
            output_path=project_root / "reports" / "model_performance.md"
        )
        
        logger.info("\n" + "="*60)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info("="*60)
        logger.info(f"\nBest Model: {best_name}")
        logger.info(f"Model saved to: {project_root / args.output_dir}")
        logger.info(f"Report saved to: {project_root / 'reports' / 'model_performance.md'}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())