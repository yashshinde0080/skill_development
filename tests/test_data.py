"""
Tests for data loading and preprocessing modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader, load_sample_data
from src.data.data_preprocessing import DataPreprocessor


class TestDataLoader:
    """Tests for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance."""
        return DataLoader()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return load_sample_data()
    
    def test_load_sample_data(self):
        """Test loading sample data."""
        df = load_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'selling_price' in df.columns
        assert 'year' in df.columns
    
    def test_validate_data_valid(self, data_loader, sample_data):
        """Test data validation with valid data."""
        data_loader._data = sample_data
        is_valid, issues = data_loader.validate_data()
        
        # Sample data should be valid
        assert is_valid or len(issues) == 0
    
    def test_validate_data_empty(self, data_loader):
        """Test data validation with empty DataFrame."""
        data_loader._data = pd.DataFrame()
        is_valid, issues = data_loader.validate_data()
        
        assert not is_valid
        assert len(issues) > 0
    
    def test_get_data_summary(self, data_loader, sample_data):
        """Test data summary generation."""
        data_loader._data = sample_data
        summary = data_loader.get_data_summary()
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'missing_values' in summary
        assert summary['shape'][0] == len(sample_data)
    
    def test_split_data(self, data_loader, sample_data):
        """Test data splitting."""
        train_df, test_df = data_loader.split_data(
            sample_data,
            test_size=0.2,
            save=False
        )
        
        total_rows = len(train_df) + len(test_df)
        assert total_rows == len(sample_data)
        assert len(test_df) > 0
        assert len(train_df) > len(test_df)


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance."""
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return load_sample_data()
    
    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning."""
        df_clean = preprocessor.clean_data(sample_data)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert len(df_clean) <= len(sample_data)
        # Should have mileage column renamed
        assert 'mileage' in df_clean.columns or 'mileage(km/ltr/kg)' not in df_clean.columns
    
    def test_extract_number(self, preprocessor):
        """Test number extraction from strings."""
        assert preprocessor._extract_number("23.4 kmpl") == 23.4
        assert preprocessor._extract_number("1248 CC") == 1248.0
        assert preprocessor._extract_number("74 bhp") == 74.0
        assert preprocessor._extract_number(100) == 100.0
        assert preprocessor._extract_number(None) is None
    
    def test_create_features(self, preprocessor, sample_data):
        """Test feature creation."""
        df_clean = preprocessor.clean_data(sample_data)
        df_features = preprocessor.create_features(df_clean)
        
        assert 'car_age' in df_features.columns
        assert df_features['car_age'].min() >= 0
    
    def test_prepare_features(self, preprocessor, sample_data):
        """Test feature preparation."""
        df_clean = preprocessor.clean_data(sample_data)
        df_features = preprocessor.create_features(df_clean)
        
        X, y = preprocessor.prepare_features(df_features, fit=True)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert X.shape[0] == len(sample_data)
    
    def test_preprocessor_transform(self, preprocessor, sample_data):
        """Test preprocessor transform without fitting."""
        df_clean = preprocessor.clean_data(sample_data)
        df_features = preprocessor.create_features(df_clean)
        
        # First fit
        X_fit, _ = preprocessor.prepare_features(df_features, fit=True)
        
        # Then transform
        X_transform, _ = preprocessor.prepare_features(df_features, fit=False)
        
        assert X_fit.shape == X_transform.shape