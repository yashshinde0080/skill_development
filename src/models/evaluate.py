"""
Model evaluation module for the car price prediction project.

This module handles comprehensive model evaluation including various metrics,
visualization, and comparison of different models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

from src.config.config_loader import get_config
from src.utils.helpers import get_project_root


class ModelEvaluator:
    """
    Model evaluator class for comprehensive model evaluation.
    
    Attributes:
        config: Configuration loader instance.
        evaluation_results: Dictionary storing evaluation results.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model evaluator.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        self.project_root = get_project_root()
        self.evaluation_results: Dict[str, Dict] = {}
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a single model with multiple metrics.
        
        Args:
            model: Trained model to evaluate.
            X_test: Test features.
            y_test: Test target.
            model_name: Name of the model for logging.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Log results
        self._log_metrics(model_name, metrics)
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of metrics.
        """
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Error
        mae = mean_absolute_error(y_true, y_pred)
        
        # R-squared Score
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        # Handle zero values in y_true
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = np.nan
        
        # Root Mean Squared Log Error (for positive values)
        if (y_true > 0).all() and (y_pred > 0).all():
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
        else:
            rmsle = np.nan
        
        # Median Absolute Error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Explained Variance
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'rmsle': rmsle,
            'median_ae': median_ae,
            'max_error': max_error,
            'explained_variance': explained_variance
        }
        
        return metrics
    
    def _log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics.
        
        Args:
            model_name: Name of the model.
            metrics: Dictionary of metrics.
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"RMSE:              {metrics['rmse']:,.2f}")
        logger.info(f"MAE:               {metrics['mae']:,.2f}")
        logger.info(f"R² Score:          {metrics['r2']:.4f}")
        logger.info(f"MAPE:              {metrics['mape']:.2f}%")
        logger.info(f"Median AE:         {metrics['median_ae']:,.2f}")
        logger.info(f"Max Error:         {metrics['max_error']:,.2f}")
        logger.info(f"Explained Variance: {metrics['explained_variance']:.4f}")
        logger.info(f"{'='*50}\n")
    
    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and compare results.
        
        Args:
            models: Dictionary of trained models.
            X_test: Test features.
            y_test: Test target.
            
        Returns:
            DataFrame with comparison of all models.
        """
        logger.info(f"Evaluating {len(models)} models")
        
        for name, model in models.items():
            self.evaluate_model(model, X_test, y_test, name)
        
        return self.get_comparison_table()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get a comparison table of all evaluated models.
        
        Returns:
            DataFrame with model comparisons.
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        comparison_data = []
        for name, results in self.evaluation_results.items():
            row = {'model': name}
            row.update(results['metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('rmse')
        
        return comparison_df
    
    def plot_predictions(
        self,
        model_name: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            model_name: Name of the model to plot. If None, plots all.
            save_path: Path to save the plot.
        """
        if model_name:
            models_to_plot = {model_name: self.evaluation_results[model_name]}
        else:
            models_to_plot = self.evaluation_results
        
        n_models = len(models_to_plot)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, results) in zip(axes, models_to_plot.items()):
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='none')
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Price', fontsize=12)
            ax.set_ylabel('Predicted Price', fontsize=12)
            ax.set_title(f'{name}\nR² = {results["metrics"]["r2"]:.4f}', fontsize=14)
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")
        
        plt.show()
    
    def plot_residuals(
        self,
        model_name: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot residual analysis for a model.
        
        Args:
            model_name: Name of the model.
            save_path: Path to save the plot.
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results.")
        
        results = self.evaluation_results[model_name]
        y_test = results['y_test']
        y_pred = results['y_pred']
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        
        # Residuals Distribution
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals vs Index (check for patterns)
        axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.5, edgecolors='none')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Index')
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved residuals plot to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        metrics: List[str] = ['rmse', 'mae', 'r2'],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot comparison of models across different metrics.
        
        Args:
            metrics: List of metrics to compare.
            save_path: Path to save the plot.
        """
        comparison_df = self.get_comparison_table()
        
        if comparison_df.empty:
            logger.warning("No evaluation results to compare.")
            return
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(comparison_df))
        
        for ax, metric in zip(axes, metrics):
            if metric not in comparison_df.columns:
                continue
            
            bars = ax.bar(comparison_df['model'], comparison_df[metric], color=colors)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, comparison_df[metric]):
                height = bar.get_height()
                ax.annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance.
            top_n: Number of top features to show.
            save_path: Path to save the plot.
        """
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("viridis", len(top_features))
        bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
        
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, top_features['importance']):
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def plot_error_distribution(
        self,
        model_name: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot error distribution analysis.
        
        Args:
            model_name: Name of the model.
            save_path: Path to save the plot.
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results.")
        
        results = self.evaluation_results[model_name]
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # Calculate percentage errors
        percentage_errors = ((y_pred - y_test) / y_test) * 100
        absolute_errors = np.abs(y_test - y_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Absolute Error Distribution
        axes[0].hist(absolute_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=np.mean(absolute_errors), color='r', linestyle='--', 
                        label=f'Mean: {np.mean(absolute_errors):,.0f}')
        axes[0].axvline(x=np.median(absolute_errors), color='g', linestyle='--',
                        label=f'Median: {np.median(absolute_errors):,.0f}')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Absolute Error Distribution')
        axes[0].legend()
        
        # Percentage Error Distribution
        axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1].axvline(x=0, color='k', linestyle='-', linewidth=2)
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Percentage Error Distribution')
        
        # Error by Price Range
        price_bins = pd.qcut(y_test, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        error_by_range = pd.DataFrame({
            'price_range': price_bins,
            'absolute_error': absolute_errors
        })
        error_summary = error_by_range.groupby('price_range')['absolute_error'].mean()
        
        axes[2].bar(error_summary.index, error_summary.values, color='teal', edgecolor='black')
        axes[2].set_xlabel('Price Range')
        axes[2].set_ylabel('Mean Absolute Error')
        axes[2].set_title('MAE by Price Range')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Error Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error distribution plot to {save_path}")
        
        plt.show()
    
    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report.
            
        Returns:
            Report as a string.
        """
        comparison_df = self.get_comparison_table()
        
        report = []
        report.append("# Model Evaluation Report\n")
        report.append("=" * 60 + "\n\n")
        
        report.append("## Summary\n")
        report.append(f"Number of models evaluated: {len(self.evaluation_results)}\n\n")
        
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            report.append(f"**Best Model:** {best_model['model']}\n")
            report.append(f"- RMSE: {best_model['rmse']:,.2f}\n")
            report.append(f"- MAE: {best_model['mae']:,.2f}\n")
            report.append(f"- R² Score: {best_model['r2']:.4f}\n\n")
        
        report.append("## Detailed Results\n\n")
        report.append("### Model Comparison Table\n\n")
        report.append(comparison_df.to_markdown(index=False) + "\n\n")
        
        for name, results in self.evaluation_results.items():
            report.append(f"### {name}\n\n")
            metrics = results['metrics']
            for metric, value in metrics.items():
                if pd.notna(value):
                    if metric in ['rmse', 'mae', 'median_ae', 'max_error']:
                        report.append(f"- {metric.upper()}: {value:,.2f}\n")
                    elif metric == 'mape':
                        report.append(f"- {metric.upper()}: {value:.2f}%\n")
                    else:
                        report.append(f"- {metric.upper()}: {value:.4f}\n")
            report.append("\n")
        
        report_text = "".join(report)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved evaluation report to {output_path}")
        
        return report_text