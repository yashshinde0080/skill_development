"""
Data loading module for the car price prediction project.

This module handles loading data from various sources including CSV files,
databases, and APIs.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.config.config_loader import get_config
from src.utils.helpers import get_project_root


class DataLoader:
    """
    Data loader class for loading and initial validation of car price data.
    
    Attributes:
        config: Configuration loader instance.
        data_path: Path to the data file.
    """
    
    # Expected columns based on the dataset specification
    EXPECTED_COLUMNS = [
        'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
        'transmission', 'owner', 'mileage(km/ltr/kg)', 'engine',
        'max_power', 'seats'
    ]
    
    # Column data types
    COLUMN_DTYPES = {
        'year': 'int64',
        'selling_price': 'int64',
        'km_driven': 'int64',
        'fuel': 'object',
        'seller_type': 'object',
        'transmission': 'object',
        'owner': 'object',
        'mileage(km/ltr/kg)': 'object',
        'engine': 'object',
        'max_power': 'object',
        'seats': 'float64'
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        self.project_root = get_project_root()
        self._data: Optional[pd.DataFrame] = None
        
        logger.info("DataLoader initialized")
    
    def load_csv(
        self,
        filepath: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file. If None, uses config path.
            **kwargs: Additional arguments to pass to pandas read_csv.
            
        Returns:
            Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the data doesn't match expected format.
        """
        if filepath is None:
            filepath = self.project_root / self.config.get('data.raw_path')
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        # Load the data
        df = pd.read_csv(filepath, **kwargs)
        
        # Log basic info
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        # Store data
        self._data = df
        
        return df
    
    def validate_data(self, df: Optional[pd.DataFrame] = None) -> Tuple[bool, List[str]]:
        """
        Validate the loaded data against expected schema.
        
        Args:
            df: DataFrame to validate. If None, uses stored data.
            
        Returns:
            Tuple of (is_valid, list of issues).
        """
        if df is None:
            df = self._data
        
        if df is None:
            return False, ["No data loaded"]
        
        issues = []
        
        # Check for required columns
        missing_columns = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check for extra columns
        extra_columns = set(df.columns) - set(self.EXPECTED_COLUMNS)
        if extra_columns:
            logger.warning(f"Extra columns found: {extra_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        # Check target column
        target = self.config.get('features.target', 'selling_price')
        if target in df.columns:
            if df[target].isnull().any():
                issues.append(f"Target column '{target}' contains null values")
            if (df[target] <= 0).any():
                logger.warning(f"Target column '{target}' contains non-positive values")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.error(f"Data validation failed: {issues}")
        
        return is_valid, issues
    
    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get a summary of the loaded data.
        
        Args:
            df: DataFrame to summarize. If None, uses stored data.
            
        Returns:
            Dictionary containing data summary.
        """
        if df is None:
            df = self._data
        
        if df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
        }
        
        # Numerical column statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary["numerical_stats"] = df[numerical_cols].describe().to_dict()
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary["categorical_stats"] = {
                col: {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
                for col in categorical_cols
            }
        
        return summary
    
    def split_data(
        self,
        df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = None,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame to split. If None, uses stored data.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
            stratify_column: Column to use for stratified splitting.
            save: Whether to save the split data.
            
        Returns:
            Tuple of (train_df, test_df).
        """
        from sklearn.model_selection import train_test_split
        
        if df is None:
            df = self._data
        
        if df is None:
            raise ValueError("No data available for splitting")
        
        logger.info(f"Splitting data with test_size={test_size}")
        
        stratify = None
        if stratify_column and stratify_column in df.columns:
            stratify = df[stratify_column]
            logger.info(f"Using stratified split on column: {stratify_column}")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
        
        if save:
            self._save_split_data(train_df, test_df)
        
        return train_df, test_df
    
    def _save_split_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        Save split data to processed directory.
        
        Args:
            train_df: Training DataFrame.
            test_df: Testing DataFrame.
        """
        processed_path = self.project_root / self.config.get('data.processed_path')
        processed_path.mkdir(parents=True, exist_ok=True)
        
        train_file = processed_path / self.config.get('data.train_file', 'train_data.csv')
        test_file = processed_path / self.config.get('data.test_file', 'test_data.csv')
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        logger.info(f"Saved train data to {train_file}")
        logger.info(f"Saved test data to {test_file}")
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed train and test data.
        
        Returns:
            Tuple of (train_df, test_df).
        """
        processed_path = self.project_root / self.config.get('data.processed_path')
        
        train_file = processed_path / self.config.get('data.train_file', 'train_data.csv')
        test_file = processed_path / self.config.get('data.test_file', 'test_data.csv')
        
        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError("Processed data files not found. Run split_data first.")
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        logger.info(f"Loaded processed data: train={len(train_df)}, test={len(test_df)}")
        
        return train_df, test_df
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the stored data."""
        return self._data
    
    @data.setter
    def data(self, df: pd.DataFrame) -> None:
        """Set the stored data."""
        self._data = df


def load_sample_data() -> pd.DataFrame:
    """
    Load sample car price data for testing purposes.
    
    Returns:
        Sample DataFrame with car data.
    """
    # Create sample data matching the expected schema
    sample_data = {
        'year': [2014, 2016, 2018, 2017, 2015, 2019, 2020, 2013, 2021, 2012],
        'selling_price': [450000, 650000, 850000, 720000, 380000, 1200000, 1500000, 290000, 1800000, 250000],
        'km_driven': [145000, 85000, 45000, 67000, 120000, 30000, 15000, 150000, 8000, 175000],
        'fuel': ['Diesel', 'Petrol', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel'],
        'seller_type': ['Individual', 'Dealer', 'Individual', 'Dealer', 'Individual', 'Dealer', 'Dealer', 'Individual', 'Dealer', 'Individual'],
        'transmission': ['Manual', 'Manual', 'Automatic', 'Manual', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Automatic', 'Manual'],
        'owner': ['First Owner', 'Second Owner', 'First Owner', 'First Owner', 'Second Owner', 'First Owner', 'First Owner', 'Third Owner', 'First Owner', 'Second Owner'],
        'mileage(km/ltr/kg)': ['23.4 kmpl', '18.2 kmpl', '16.5 kmpl', '21.0 kmpl', '19.5 kmpl', '22.8 kmpl', '17.0 kmpl', '24.1 kmpl', '15.2 kmpl', '20.5 kmpl'],
        'engine': ['1248 CC', '1497 CC', '1998 CC', '1493 CC', '1197 CC', '1956 CC', '2487 CC', '1396 CC', '2998 CC', '1461 CC'],
        'max_power': ['74 bhp', '117 bhp', '188 bhp', '108 bhp', '81 bhp', '167 bhp', '175 bhp', '89 bhp', '255 bhp', '63 bhp'],
        'seats': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.0, 5.0, 5.0, 5.0]
    }
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Created sample data with {len(df)} rows")
    
    return df