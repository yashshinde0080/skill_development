"""
Data preprocessing module for the car price prediction project.

This module handles data cleaning, transformation, and preparation for modeling.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

from src.config.config_loader import get_config
from src.utils.helpers import get_project_root, save_pickle


class DataPreprocessor:
    """
    Data preprocessor class for cleaning and transforming car price data.
    
    Attributes:
        config: Configuration loader instance.
        preprocessor: Fitted preprocessor pipeline.
        label_encoders: Dictionary of label encoders for categorical columns.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        self.project_root = get_project_root()
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._feature_names: List[str] = []
        
        logger.info("DataPreprocessor initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, removing duplicates,
        and fixing data types.
        
        Args:
            df: Raw DataFrame to clean.
            
        Returns:
            Cleaned DataFrame.
        """
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Clean numerical columns with units
        df_clean = self._clean_numerical_with_units(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove outliers if configured
        if self.config.get('preprocessing.outlier_removal', False):
            df_clean = self._remove_outliers(df_clean)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        
        return df_clean
    
    def _clean_numerical_with_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numerical columns that contain units (e.g., '23.4 kmpl', '1248 CC').
        
        Args:
            df: DataFrame with raw data.
            
        Returns:
            DataFrame with cleaned numerical columns.
        """
        df_clean = df.copy()
        
        # Clean mileage column (e.g., '23.4 kmpl' -> 23.4)
        if 'mileage(km/ltr/kg)' in df_clean.columns:
            df_clean['mileage'] = df_clean['mileage(km/ltr/kg)'].apply(self._extract_number)
            df_clean = df_clean.drop('mileage(km/ltr/kg)', axis=1)
            logger.debug("Cleaned mileage column")
        
        # Clean engine column (e.g., '1248 CC' -> 1248)
        if 'engine' in df_clean.columns:
            df_clean['engine'] = df_clean['engine'].apply(self._extract_number)
            logger.debug("Cleaned engine column")
        
        # Clean max_power column (e.g., '74 bhp' -> 74)
        if 'max_power' in df_clean.columns:
            df_clean['max_power'] = df_clean['max_power'].apply(self._extract_number)
            logger.debug("Cleaned max_power column")
        
        return df_clean
    
    @staticmethod
    def _extract_number(value) -> Optional[float]:
        """
        Extract numerical value from a string with units.
        
        Args:
            value: String value potentially containing numbers and units.
            
        Returns:
            Extracted float value or None if extraction fails.
        """
        if pd.isna(value):
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Find all numbers (including decimals) in the string
        matches = re.findall(r'[\d.]+', str(value))
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                return None
        
        return None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on configuration.
        
        Args:
            df: DataFrame with potential missing values.
            
        Returns:
            DataFrame with handled missing values.
        """
        if not self.config.get('preprocessing.handle_missing', True):
            return df
        
        df_clean = df.copy()
        
        # Get strategies from config
        num_strategy = self.config.get('preprocessing.missing_strategy.numerical', 'median')
        cat_strategy = self.config.get('preprocessing.missing_strategy.categorical', 'mode')
        
        # Handle numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                if num_strategy == 'median':
                    fill_value = df_clean[col].median()
                elif num_strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif num_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = df_clean[col].median()
                
                null_count = df_clean[col].isnull().sum()
                df_clean[col].fillna(fill_value, inplace=True)
                logger.debug(f"Filled {null_count} missing values in {col} with {fill_value:.2f}")
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                if cat_strategy == 'mode':
                    fill_value = df_clean[col].mode()[0]
                elif cat_strategy == 'unknown':
                    fill_value = 'Unknown'
                else:
                    fill_value = df_clean[col].mode()[0]
                
                null_count = df_clean[col].isnull().sum()
                df_clean[col].fillna(fill_value, inplace=True)
                logger.debug(f"Filled {null_count} missing values in {col} with '{fill_value}'")
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: DataFrame with potential outliers.
            
        Returns:
            DataFrame with outliers removed.
        """
        df_clean = df.copy()
        
        method = self.config.get('preprocessing.outlier_method', 'iqr')
        threshold = self.config.get('preprocessing.outlier_threshold', 1.5)
        
        if method != 'iqr':
            logger.warning(f"Unsupported outlier method: {method}. Using IQR.")
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        target = self.config.get('features.target', 'selling_price')
        
        # Only remove outliers from feature columns, not target
        feature_cols = [col for col in numerical_cols if col != target]
        
        initial_rows = len(df_clean)
        
        for col in feature_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Create mask for valid values
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
        
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing data.
        
        Args:
            df: Cleaned DataFrame.
            
        Returns:
            DataFrame with new features.
        """
        df_features = df.copy()
        
        # Create car age feature
        if self.config.get('features.create_age', True):
            current_year = self.config.get('features.current_year', 2024)
            if 'year' in df_features.columns:
                df_features['car_age'] = current_year - df_features['year']
                logger.debug(f"Created car_age feature (current_year={current_year})")
        
        # Log transform target if configured
        if self.config.get('features.log_transform_target', False):
            target = self.config.get('features.target', 'selling_price')
            if target in df_features.columns:
                df_features[f'{target}_log'] = np.log1p(df_features[target])
                logger.debug(f"Created log-transformed target: {target}_log")
        
        logger.info(f"Feature creation completed. New shape: {df_features.shape}")
        
        return df_features
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training/prediction.
        
        Args:
            df: DataFrame with features.
            fit: Whether to fit the preprocessor (True for training, False for prediction).
            
        Returns:
            Tuple of (X features array, y target array).
        """
        # Get feature and target columns from config
        target = self.config.get('features.target', 'selling_price')
        numerical_features = self.config.get('features.numerical', [])
        categorical_features = self.config.get('features.categorical', [])
        
        # Add car_age if created
        if 'car_age' in df.columns and 'car_age' not in numerical_features:
            numerical_features = numerical_features + ['car_age']
        
        # Filter to existing columns
        numerical_features = [col for col in numerical_features if col in df.columns]
        categorical_features = [col for col in categorical_features if col in df.columns]
        
        # Update mileage column name if renamed
        numerical_features = ['mileage' if col == 'mileage(km/ltr/kg)' else col 
                             for col in numerical_features]
        numerical_features = [col for col in numerical_features if col in df.columns]
        
        self._feature_names = numerical_features + categorical_features
        
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        # Extract features and target
        X = df[numerical_features + categorical_features].copy()
        y = df[target].values if target in df.columns else None
        
        if fit:
            # Create preprocessing pipeline
            self.preprocessor = self._create_preprocessor(
                numerical_features,
                categorical_features
            )
            
            # Fit and transform
            X_processed = self.preprocessor.fit_transform(X)
            logger.info(f"Preprocessor fitted. Output shape: {X_processed.shape}")
        else:
            if self.preprocessor is None:
                raise ValueError("Preprocessor not fitted. Call prepare_features with fit=True first.")
            
            X_processed = self.preprocessor.transform(X)
        
        return X_processed, y
    
    def _create_preprocessor(
        self,
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """
        Create a preprocessing pipeline.
        
        Args:
            numerical_features: List of numerical feature names.
            categorical_features: List of categorical feature names.
            
        Returns:
            ColumnTransformer preprocessor.
        """
        scaling_method = self.config.get('preprocessing.scaling_method', 'standard')
        
        # Choose scaler based on config
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Numerical pipeline
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_pipeline, numerical_features),
                ('categorical', categorical_pipeline, categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Returns:
            List of feature names.
        """
        if self.preprocessor is None:
            return self._feature_names
        
        try:
            return self.preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            return self._feature_names
    
    def save_preprocessor(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Save the fitted preprocessor.
        
        Args:
            filepath: Path to save the preprocessor.
        """
        if self.preprocessor is None:
            raise ValueError("No preprocessor to save. Fit the preprocessor first.")
        
        if filepath is None:
            filepath = self.project_root / self.config.get('model_saving.path') / 'preprocessor.pkl'
        
        save_pickle(self.preprocessor, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: Union[str, Path]) -> None:
        """
        Load a saved preprocessor.
        
        Args:
            filepath: Path to the saved preprocessor.
        """
        from src.utils.helpers import load_pickle
        
        self.preprocessor = load_pickle(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
    
    def get_preprocessing_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get a summary of preprocessing steps and their effects.
        
        Args:
            df: Original DataFrame.
            
        Returns:
            Dictionary with preprocessing summary.
        """
        summary = {
            "original_shape": df.shape,
            "missing_values_before": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }
        
        # Clean and get stats
        df_clean = self.clean_data(df)
        summary["cleaned_shape"] = df_clean.shape
        summary["missing_values_after"] = df_clean.isnull().sum().to_dict()
        
        # Feature creation
        df_features = self.create_features(df_clean)
        summary["features_shape"] = df_features.shape
        summary["new_features"] = list(set(df_features.columns) - set(df.columns))
        
        return summary