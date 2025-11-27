"""
Model prediction module for the car price prediction project.

This module handles making predictions using trained models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.config.config_loader import get_config
from src.utils.helpers import get_project_root, load_pickle
from src.data.data_preprocessing import DataPreprocessor


class ModelPredictor:
    """
    Model predictor class for making predictions with trained models.
    
    Attributes:
        config: Configuration loader instance.
        model: Loaded prediction model.
        preprocessor: Data preprocessor.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the model predictor.
        
        Args:
            model_path: Path to the saved model.
            preprocessor_path: Path to the saved preprocessor.
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        self.project_root = get_project_root()
        self.model: Optional[Any] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self._preprocessor_obj = None
        
        # Load model and preprocessor if paths provided
        if model_path:
            self.load_model(model_path)
        
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
        
        logger.info("ModelPredictor initialized")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            # Try default path
            default_path = self.project_root / self.config.get('model_saving.path') / self.config.get('model_saving.best_model_name')
            if default_path.exists():
                model_path = default_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = load_pickle(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def load_preprocessor(self, preprocessor_path: Union[str, Path]) -> None:
        """
        Load a saved preprocessor.
        
        Args:
            preprocessor_path: Path to the saved preprocessor.
        """
        preprocessor_path = Path(preprocessor_path)
        
        if not preprocessor_path.exists():
            # Try default path
            default_path = self.project_root / self.config.get('model_saving.path') / 'preprocessor.pkl'
            if default_path.exists():
                preprocessor_path = default_path
            else:
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        self._preprocessor_obj = load_pickle(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        preprocess: bool = True
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts).
            preprocess: Whether to preprocess the data.
            
        Returns:
            Array of predictions.
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        logger.info(f"Making predictions for {len(df)} samples")
        
        if preprocess:
            X = self._preprocess_data(df)
        else:
            X = df.values if isinstance(df, pd.DataFrame) else df
        
        predictions = self.model.predict(X)
        
        logger.info(f"Predictions completed")
        
        return predictions
    
    def _preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for prediction.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Preprocessed feature array.
        """
        # Initialize preprocessor if needed
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor()
        
        # Clean data
        df_clean = self.preprocessor.clean_data(df)
        
        # Create features
        df_features = self.preprocessor.create_features(df_clean)
        
        # Transform using loaded preprocessor
        if self._preprocessor_obj is not None:
            # Get feature columns
            numerical_features = self.config.get('features.numerical', [])
            categorical_features = self.config.get('features.categorical', [])
            
            # Update mileage column name if renamed
            numerical_features = ['mileage' if col == 'mileage(km/ltr/kg)' else col 
                                 for col in numerical_features]
            
            # Add car_age if created
            if 'car_age' in df_features.columns and 'car_age' not in numerical_features:
                numerical_features = numerical_features + ['car_age']
            
            # Filter to existing columns
            numerical_features = [col for col in numerical_features if col in df_features.columns]
            categorical_features = [col for col in categorical_features if col in df_features.columns]
            
            X = df_features[numerical_features + categorical_features]
            X_processed = self._preprocessor_obj.transform(X)
            
            return X_processed
        else:
            # Use preprocessor's prepare_features method
            X_processed, _ = self.preprocessor.prepare_features(df_features, fit=False)
            return X_processed
    
    def predict_single(
        self,
        year: int,
        km_driven: int,
        fuel: str,
        seller_type: str,
        transmission: str,
        owner: str,
        mileage: float,
        engine: float,
        max_power: float,
        seats: int
    ) -> float:
        """
        Make prediction for a single car.
        
        Args:
            year: Year of manufacture.
            km_driven: Kilometers driven.
            fuel: Fuel type.
            seller_type: Type of seller.
            transmission: Transmission type.
            owner: Owner type.
            mileage: Mileage (km/l or km/kg).
            engine: Engine capacity (CC).
            max_power: Maximum power (bhp).
            seats: Number of seats.
            
        Returns:
            Predicted selling price.
        """
        data = {
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }
        
        prediction = self.predict(data)
        
        return float(prediction[0])
    
    def predict_with_confidence(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        n_estimators: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals (for ensemble models).
        
        Args:
            data: Input data.
            n_estimators: Number of estimators for uncertainty estimation.
            
        Returns:
            Dictionary with predictions and confidence intervals.
        """
        predictions = self.predict(data)
        
        # Check if model supports prediction intervals
        if hasattr(self.model, 'estimators_'):
            # For ensemble models, get predictions from all estimators
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            X = self._preprocess_data(df)
            
            # Get predictions from all trees
            all_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            
            # Calculate statistics
            mean_pred = np.mean(all_predictions, axis=0)
            std_pred = np.std(all_predictions, axis=0)
            lower_bound = np.percentile(all_predictions, 2.5, axis=0)
            upper_bound = np.percentile(all_predictions, 97.5, axis=0)
            
            return {
                'prediction': mean_pred,
                'std': std_pred,
                'lower_95': lower_bound,
                'upper_95': upper_bound
            }
        else:
            # For non-ensemble models, return just predictions
            return {
                'prediction': predictions,
                'std': np.zeros_like(predictions),
                'lower_95': predictions,
                'upper_95': predictions
            }
    
    def batch_predict(
        self,
        filepath: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Make predictions on a batch of data from file.
        
        Args:
            filepath: Path to input CSV file.
            output_path: Path to save predictions.
            
        Returns:
            DataFrame with predictions.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        
        # Make predictions
        predictions = self.predict(df)
        
        # Add predictions to DataFrame
        df['predicted_price'] = predictions
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
        
        return df
    
    def get_prediction_explanation(
        self,
        data: Dict,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Get explanation for a single prediction (feature contributions).
        
        Args:
            data: Input data as dictionary.
            feature_names: List of feature names.
            
        Returns:
            Dictionary with prediction explanation.
        """
        prediction = self.predict_single(**data) if 'year' in data else self.predict(data)[0]
        
        explanation = {
            'prediction': prediction,
            'input_features': data,
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_') and feature_names:
            importances = self.model.feature_importances_
            feature_importance = dict(zip(feature_names[:len(importances)], importances))
            explanation['feature_importance'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        return explanation


def quick_predict(
    year: int,
    km_driven: int,
    fuel: str,
    seller_type: str,
    transmission: str,
    owner: str,
    mileage: float,
    engine: float,
    max_power: float,
    seats: int,
    model_path: Optional[str] = None
) -> float:
    """
    Quick prediction function for single car.
    
    Args:
        year: Year of manufacture.
        km_driven: Kilometers driven.
        fuel: Fuel type.
        seller_type: Type of seller.
        transmission: Transmission type.
        owner: Owner type.
        mileage: Mileage (km/l or km/kg).
        engine: Engine capacity (CC).
        max_power: Maximum power (bhp).
        seats: Number of seats.
        model_path: Optional path to model.
        
    Returns:
        Predicted price.
    """
    predictor = ModelPredictor(model_path=model_path)
    
    return predictor.predict_single(
        year=year,
        km_driven=km_driven,
        fuel=fuel,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner,
        mileage=mileage,
        engine=engine,
        max_power=max_power,
        seats=seats
    )