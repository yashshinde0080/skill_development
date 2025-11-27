"""
Feature engineering module for the car price prediction project.

This module handles advanced feature engineering and selection.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression,
    mutual_info_regression
)
from sklearn.ensemble import RandomForestRegressor

from src.config.config_loader import get_config


class FeatureBuilder:
    """
    Feature builder class for creating and selecting features.
    
    Attributes:
        config: Configuration loader instance.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature builder.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        self._important_features: List[str] = []
        
        logger.info("FeatureBuilder initialized")
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build additional features from existing data.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with additional features.
        """
        df_features = df.copy()
        
        # Age-based features
        df_features = self._create_age_features(df_features)
        
        # Mileage-based features
        df_features = self._create_mileage_features(df_features)
        
        # Power-related features
        df_features = self._create_power_features(df_features)
        
        # Interaction features
        df_features = self._create_interaction_features(df_features)
        
        # Categorical encoding features
        df_features = self._create_categorical_features(df_features)
        
        logger.info(f"Feature building completed. Shape: {df_features.shape}")
        
        return df_features
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with age features.
        """
        df_out = df.copy()
        current_year = self.config.get('features.current_year', 2024)
        
        if 'year' in df_out.columns:
            # Car age
            df_out['car_age'] = current_year - df_out['year']
            
            # Age categories
            df_out['age_category'] = pd.cut(
                df_out['car_age'],
                bins=[0, 2, 5, 10, 20, float('inf')],
                labels=['new', 'recent', 'moderate', 'old', 'very_old']
            )
            
            # Age squared (for non-linear relationships)
            df_out['car_age_squared'] = df_out['car_age'] ** 2
            
            logger.debug("Created age-related features")
        
        return df_out
    
    def _create_mileage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mileage-related features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with mileage features.
        """
        df_out = df.copy()
        
        if 'km_driven' in df_out.columns:
            # Log of km_driven
            df_out['km_driven_log'] = np.log1p(df_out['km_driven'])
            
            # Km driven categories
            df_out['km_category'] = pd.cut(
                df_out['km_driven'],
                bins=[0, 30000, 60000, 100000, 150000, float('inf')],
                labels=['low', 'moderate', 'medium', 'high', 'very_high']
            )
            
            # Average yearly km driven
            if 'car_age' in df_out.columns:
                df_out['avg_yearly_km'] = df_out['km_driven'] / (df_out['car_age'] + 1)
            
            logger.debug("Created mileage-related features")
        
        return df_out
    
    def _create_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create power-related features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with power features.
        """
        df_out = df.copy()
        
        # Power to weight ratio (using engine as proxy for weight)
        if 'max_power' in df_out.columns and 'engine' in df_out.columns:
            df_out['power_per_cc'] = df_out['max_power'] / (df_out['engine'] + 1)
            logger.debug("Created power_per_cc feature")
        
        # Power categories
        if 'max_power' in df_out.columns:
            df_out['power_category'] = pd.cut(
                df_out['max_power'],
                bins=[0, 70, 100, 150, 200, float('inf')],
                labels=['low', 'economy', 'moderate', 'high', 'premium']
            )
        
        # Engine size categories
        if 'engine' in df_out.columns:
            df_out['engine_category'] = pd.cut(
                df_out['engine'],
                bins=[0, 1000, 1500, 2000, 2500, float('inf')],
                labels=['small', 'compact', 'medium', 'large', 'premium']
            )
        
        return df_out
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with interaction features.
        """
        df_out = df.copy()
        
        # Age and km interaction
        if 'car_age' in df_out.columns and 'km_driven' in df_out.columns:
            df_out['age_km_interaction'] = df_out['car_age'] * df_out['km_driven'] / 100000
        
        # Power efficiency score (mileage * power)
        if 'mileage' in df_out.columns and 'max_power' in df_out.columns:
            df_out['efficiency_score'] = df_out['mileage'] * df_out['max_power'] / 100
        
        logger.debug("Created interaction features")
        
        return df_out
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create encoded categorical features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with categorical features.
        """
        df_out = df.copy()
        
        # Owner rank (ordinal encoding)
        if 'owner' in df_out.columns:
            owner_mapping = {
                'First Owner': 1,
                'Second Owner': 2,
                'Third Owner': 3,
                'Fourth & Above Owner': 4,
                'Test Drive Car': 0
            }
            df_out['owner_rank'] = df_out['owner'].map(owner_mapping).fillna(2)
        
        # Binary features
        if 'transmission' in df_out.columns:
            df_out['is_automatic'] = (df_out['transmission'] == 'Automatic').astype(int)
        
        if 'fuel' in df_out.columns:
            df_out['is_diesel'] = (df_out['fuel'] == 'Diesel').astype(int)
            df_out['is_petrol'] = (df_out['fuel'] == 'Petrol').astype(int)
        
        if 'seller_type' in df_out.columns:
            df_out['is_dealer'] = (df_out['seller_type'] == 'Dealer').astype(int)
        
        logger.debug("Created categorical features")
        
        return df_out
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str = 'mutual_info',
        n_features: int = 10
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select top features based on importance.
        
        Args:
            X: Feature matrix.
            y: Target array.
            feature_names: List of feature names.
            method: Selection method ('mutual_info', 'f_regression', 'rfe').
            n_features: Number of features to select.
            
        Returns:
            Tuple of (selected features array, selected feature names).
        """
        n_features = min(n_features, X.shape[1])
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            mask = selector.get_support()
            selected_names = [name for name, selected in zip(feature_names, mask) if selected]
        else:
            selected_names = feature_names[:n_features]
        
        self._important_features = selected_names
        
        logger.info(f"Selected {len(selected_names)} features using {method}")
        logger.debug(f"Selected features: {selected_names}")
        
        return X_selected, selected_names
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X: Feature matrix.
            y: Target array.
            feature_names: List of feature names.
            
        Returns:
            DataFrame with feature importance scores.
        """
        # Train a random forest to get importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names[:X.shape[1]],
            'importance': rf.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        logger.info("Feature importance calculated")
        
        return importance_df
    
    def analyze_correlations(
        self,
        df: pd.DataFrame,
        threshold: float = 0.8
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Analyze correlations between numerical features.
        
        Args:
            df: DataFrame with numerical features.
            threshold: Correlation threshold for identifying high correlations.
            
        Returns:
            Dictionary with correlation analysis results.
        """
        numerical_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numerical_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    high_correlations.append((col1, col2, corr))
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        target = self.config.get('features.target', 'selling_price')
        target_correlations = []
        
        if target in correlation_matrix.columns:
            for col in correlation_matrix.columns:
                if col != target:
                    corr = correlation_matrix.loc[target, col]
                    target_correlations.append((col, corr))
            
            target_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        results = {
            'high_correlations': high_correlations,
            'target_correlations': target_correlations
        }
        
        logger.info(f"Found {len(high_correlations)} highly correlated feature pairs")
        
        return results
    
    @property
    def important_features(self) -> List[str]:
        """Get the list of important features."""
        return self._important_features