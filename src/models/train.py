"""
Model training module for the car price prediction project.

This module handles training of various machine learning models
including Linear Regression, Random Forest, and XGBoost.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import xgboost as xgb

from src.config.config_loader import get_config
from src.utils.helpers import get_project_root, save_pickle, timer_decorator


class ModelTrainer:
    """
    Model trainer class for training and tuning ML models.
    
    Attributes:
        config: Configuration loader instance.
        models: Dictionary of trained models.
        best_model: Best performing model.
    """
    
    AVAILABLE_MODELS = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'xgboost': xgb.XGBRegressor
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        self.project_root = get_project_root()
        self.models: Dict[str, Any] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.training_results: Dict[str, Dict] = {}
        
        logger.info("ModelTrainer initialized")
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune_hyperparameters: bool = False
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            tune_hyperparameters: Whether to perform hyperparameter tuning.
            
        Returns:
            Dictionary of trained models.
        """
        logger.info("Starting training for all configured models")
        
        # Train Linear Regression
        if self.config.get('models.linear_regression.enabled', True):
            self._train_linear_regression(X_train, y_train)
        
        # Train Random Forest
        if self.config.get('models.random_forest.enabled', True):
            self._train_random_forest(X_train, y_train, tune_hyperparameters)
        
        # Train XGBoost
        if self.config.get('models.xgboost.enabled', True):
            self._train_xgboost(X_train, y_train, tune_hyperparameters)
        
        logger.info(f"Completed training {len(self.models)} models")
        
        return self.models
    
    @timer_decorator
    def _train_linear_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> LinearRegression:
        """
        Train Linear Regression model.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Trained Linear Regression model.
        """
        logger.info("Training Linear Regression model")
        
        params = self.config.get('models.linear_regression.params', {})
        model = LinearRegression(**params)
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = self._cross_validate(model, X_train, y_train)
        
        self.models['linear_regression'] = model
        self.training_results['linear_regression'] = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
        
        logger.info(f"Linear Regression - CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return model
    
    @timer_decorator
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = False
    ) -> RandomForestRegressor:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            tune: Whether to tune hyperparameters.
            
        Returns:
            Trained Random Forest model.
        """
        logger.info("Training Random Forest model")
        
        params = self.config.get('models.random_forest.params', {})
        
        if tune and self.config.get('hyperparameter_tuning.enabled', False):
            model = self._tune_random_forest(X_train, y_train)
        else:
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = self._cross_validate(model, X_train, y_train)
        
        self.models['random_forest'] = model
        self.training_results['random_forest'] = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'best_params': model.get_params() if not tune else getattr(model, 'best_params_', params)
        }
        
        logger.info(f"Random Forest - CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return model
    
    def _tune_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestRegressor:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Best tuned Random Forest model.
        """
        logger.info("Tuning Random Forest hyperparameters")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        cv_folds = self.config.get('hyperparameter_tuning.cv_folds', 5)
        scoring = self.config.get('hyperparameter_tuning.scoring', 'neg_root_mean_squared_error')
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best RF parameters: {grid_search.best_params_}")
        logger.info(f"Best RF score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    @timer_decorator
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        tune: bool = False
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            tune: Whether to tune hyperparameters.
            
        Returns:
            Trained XGBoost model.
        """
        logger.info("Training XGBoost model")
        
        params = self.config.get('models.xgboost.params', {})
        
        if tune and self.config.get('hyperparameter_tuning.enabled', False):
            model = self._tune_xgboost(X_train, y_train)
        else:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
        
        # Cross-validation score
        cv_scores = self._cross_validate(model, X_train, y_train)
        
        self.models['xgboost'] = model
        self.training_results['xgboost'] = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'best_params': model.get_params() if not tune else getattr(model, 'best_params_', params)
        }
        
        logger.info(f"XGBoost - CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return model
    
        def _tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> xgb.XGBRegressor:
            """
        Tune XGBoost hyperparameters.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            
        Returns:
            Best tuned XGBoost model.
        """
        logger.info("Tuning XGBoost hyperparameters")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        cv_folds = self.config.get('hyperparameter_tuning.cv_folds', 5)
        scoring = self.config.get('hyperparameter_tuning.scoring', 'neg_root_mean_squared_error')
        
        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best XGBoost parameters: {random_search.best_params_}")
        logger.info(f"Best XGBoost score: {-random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def _cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> np.ndarray:
        """
        Perform cross-validation for a model.
        
        Args:
            model: Model to evaluate.
            X: Features.
            y: Target.
            cv: Number of cross-validation folds.
            
        Returns:
            Array of cross-validation scores.
        """
        cv_folds = self.config.get('evaluation.cross_validation.folds', cv)
        scoring = 'neg_root_mean_squared_error'
        
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        # Convert negative RMSE to positive
        return -scores
    
    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict] = None,
        tune: bool = False
    ) -> Any:
        """
        Train a single model by name.
        
        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training target.
            params: Optional model parameters.
            tune: Whether to tune hyperparameters.
            
        Returns:
            Trained model.
            
        Raises:
            ValueError: If model_name is not recognized.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        logger.info(f"Training {model_name}")
        
        # Get parameters from config if not provided
        if params is None:
            params = self.config.get(f'models.{model_name}.params', {})
        
        # Create and train model
        model_class = self.AVAILABLE_MODELS[model_name]
        model = model_class(**params)
        
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = self._cross_validate(model, X_train, y_train)
        
        self.models[model_name] = model
        self.training_results[model_name] = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'params': params
        }
        
        logger.info(f"{model_name} - CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return model
    
    def select_best_model(
        self,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        metric: str = 'cv_score'
    ) -> Tuple[str, Any]:
        """
        Select the best performing model based on validation performance.
        
        Args:
            X_val: Optional validation features.
            y_val: Optional validation target.
            metric: Metric to use for selection ('cv_score' or 'val_score').
            
        Returns:
            Tuple of (best model name, best model).
        """
        if not self.models:
            raise ValueError("No models trained yet. Train models first.")
        
        best_score = float('inf')
        best_name = None
        
        for name, model in self.models.items():
            if metric == 'cv_score':
                score = self.training_results[name]['mean_cv_score']
            elif metric == 'val_score' and X_val is not None and y_val is not None:
                from sklearn.metrics import mean_squared_error
                y_pred = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                score = self.training_results[name]['mean_cv_score']
            
            if score < best_score:
                best_score = score
                best_name = name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        logger.info(f"Best model: {best_name} with RMSE: {best_score:.4f}")
        
        return best_name, self.best_model
    
    def save_models(
        self,
        save_all: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Save trained models to disk.
        
        Args:
            save_all: Whether to save all models or just the best.
            save_path: Path to save models.
        """
        if save_path is None:
            save_path = self.project_root / self.config.get('model_saving.path', 'models/')
        else:
            save_path = Path(save_path)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        if save_all:
            for name, model in self.models.items():
                model_path = save_path / f"{name}_model.pkl"
                save_pickle(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save best model
        if self.best_model is not None:
            best_model_name = self.config.get('model_saving.best_model_name', 'best_model.pkl')
            best_model_path = save_path / best_model_name
            save_pickle(self.best_model, best_model_path)
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
        
        # Save training results
        results_path = save_path / 'training_results.pkl'
        save_pickle(self.training_results, results_path)
        logger.info(f"Saved training results to {results_path}")
    
    def load_model(
        self,
        model_name: str,
        model_path: Optional[Union[str, Path]] = None
    ) -> Any:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load.
            model_path: Path to the model file.
            
        Returns:
            Loaded model.
        """
        from src.utils.helpers import load_pickle
        
        if model_path is None:
            model_path = self.project_root / self.config.get('model_saving.path') / f"{model_name}_model.pkl"
        
        model = load_pickle(model_path)
        self.models[model_name] = model
        
        logger.info(f"Loaded {model_name} model from {model_path}")
        
        return model
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary of training results for all models.
        
        Returns:
            DataFrame with training summary.
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        for name, results in self.training_results.items():
            summary_data.append({
                'model': name,
                'mean_cv_rmse': results['mean_cv_score'],
                'std_cv_rmse': results['std_cv_score'],
                'min_cv_rmse': np.min(results['cv_scores']),
                'max_cv_rmse': np.max(results['cv_scores'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('mean_cv_rmse')
        
        return summary_df
    
    def get_feature_importance(
        self,
        model_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model. If None, uses best model.
            feature_names: List of feature names.
            
        Returns:
            DataFrame with feature importance.
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Run select_best_model first.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found.")
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        return importance_df