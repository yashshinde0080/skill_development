"""
Tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_builder import FeatureBuilder
from src.data.data_loader import load_sample_data
from src.data.data_preprocessing import DataPreprocessor


class TestFeatureBuilder:
    """Tests for FeatureBuilder class."""
    
    @pytest.fixture
    def feature_builder(self):
        """Create FeatureBuilder instance."""
        return FeatureBuilder()
    
    @pytest.fixture
    def prepared_data(self):
        """Create prepared data for testing."""
        preprocessor = DataPreprocessor()
        df = load_sample_data()
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        return df_features
    
    def test_build_features(self, feature_builder, prepared_data):
        """Test feature building."""
        df_with_features = feature_builder.build_features(prepared_data)
        
        assert isinstance(df_with_features, pd.DataFrame)
        assert len(df_with_features) == len(prepared_data)
        # Should have more columns than original
        assert len(df_with_features.columns) >= len(prepared_data.columns)
    
    def test_create_age_features(self, feature_builder, prepared_data):
        """Test age feature creation."""
        df = feature_builder._create_age_features(prepared_data)
        
        assert 'car_age' in df.columns
        if 'age_category' in df.columns:
            assert df['age_category'].dtype.name == 'category'
    
    def test_create_mileage_features(self, feature_builder, prepared_data):
        """Test mileage feature creation."""
        # Ensure car_age exists first
        prepared_data['car_age'] = 2024 - prepared_data['year']
        
        df = feature_builder._create_mileage_features(prepared_data)
        
        if 'km_driven' in df.columns:
            assert 'km_driven_log' in df.columns
    
    def test_create_categorical_features(self, feature_builder, prepared_data):
        """Test categorical feature creation."""
        df = feature_builder._create_categorical_features(prepared_data)
        
        if 'owner' in prepared_data.columns:
            assert 'owner_rank' in df.columns
        
        if 'transmission' in prepared_data.columns:
            assert 'is_automatic' in df.columns
    
    def test_analyze_correlations(self, feature_builder, prepared_data):
        """Test correlation analysis."""
        correlations = feature_builder.analyze_correlations(prepared_data)
        
        assert 'high_correlations' in correlations
        assert 'target_correlations' in correlations
        assert isinstance(correlations['target_correlations'], list)
    
    def test_get_feature_importance(self, feature_builder, prepared_data):
        """Test feature importance calculation."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(prepared_data, fit=True)
        feature_names = preprocessor.get_feature_names()
        
        importance_df = feature_builder.get_feature_importance(
            X, y, feature_names
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert importance_df['importance'].sum() == pytest.approx(1.0, rel=0.01)