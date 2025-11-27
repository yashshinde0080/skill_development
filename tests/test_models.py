"""
Tests for model training, evaluation, and prediction modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import ModelPredictor
from src.data.data_loader import load_sample_data
from src.data.data_preprocessing import DataPreprocessor


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance."""
        return ModelTrainer()
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        preprocessor = DataPreprocessor()
        df = load_sample_data()
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        X, y = preprocessor.prepare_features(df_features, fit=True)
        return X, y, preprocessor
    
    def test_train_single_model_linear_regression(self, trainer, training_data):
        """Test training a linear regression model."""
        X, y, _ = training_data
        
        model = trainer.train_single_model(
            'linear_regression',
            X, y
        )
        
        assert model is not None
        assert 'linear_regression' in trainer.models
        assert hasattr(model, 'predict')
    
    def test_train_single_model_random_forest(self, trainer, training_data):
        """Test training a random forest model."""
        X, y, _ = training_data
        
        model = trainer.train_single_model(
            'random_forest',
            X, y,
            params={'n_estimators': 10, 'random_state': 42}
        )
        
        assert model is not None
        assert 'random_forest' in trainer.models
        assert hasattr(model, 'feature_importances_')
    
    def test_train_all_models(self, trainer, training_data):
        """Test training all models."""
        X, y, _ = training_data
        
        models = trainer.train_all_models(X, y, tune_hyperparameters=False)
        
        assert len(models) >= 1
        assert len(trainer.training_results) >= 1
    
    def test_select_best_model(self, trainer, training_data):
        """Test best model selection."""
        X, y, _ = training_data
        
        trainer.train_all_models(X, y, tune_hyperparameters=False)
        best_name, best_model = trainer.select_best_model()
        
        assert best_name is not None
        assert best_model is not None
        assert trainer.best_model_name in trainer.models
    
    def test_get_training_summary(self, trainer, training_data):
        """Test training summary generation."""
        X, y, _ = training_data
        
        trainer.train_all_models(X, y, tune_hyperparameters=False)
        summary = trainer.get_training_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'model' in summary.columns
        assert 'mean_cv_rmse' in summary.columns
    
    def test_save_and_load_model(self, trainer, training_data):
        """Test model saving and loading."""
        X, y, _ = training_data
        
        trainer.train_single_model('linear_regression', X, y)
        trainer.best_model = trainer.models['linear_regression']
        trainer.best_model_name = 'linear_regression'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_models(save_all=True, save_path=tmpdir)
            
            # Check if model file exists
        def test_save_and_load_model(self, trainer, training_data):
         """Test model saving and loading."""
        X, y, _ = training_data
        
        trainer.train_single_model('linear_regression', X, y)
        trainer.best_model = trainer.models['linear_regression']
        trainer.best_model_name = 'linear_regression'
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_models(save_all=True, save_path=tmpdir)
            
            # Check files exist
            assert Path(tmpdir, 'linear_regression_model.pkl').exists()
            assert Path(tmpdir, 'best_model.pkl').exists()
            
            # Load model
            new_trainer = ModelTrainer()
            loaded_model = new_trainer.load_model(
                'linear_regression',
                model_path=Path(tmpdir, 'linear_regression_model.pkl')
            )
            
            assert loaded_model is not None
            
            # Test predictions match
            predictions_original = trainer.models['linear_regression'].predict(X)
            predictions_loaded = loaded_model.predict(X)
            
            np.testing.assert_array_almost_equal(
                predictions_original,
                predictions_loaded
            )
    
    def test_get_feature_importance(self, trainer, training_data):
        """Test feature importance extraction."""
        X, y, preprocessor = training_data
        
        trainer.train_single_model(
            'random_forest',
            X, y,
            params={'n_estimators': 10, 'random_state': 42}
        )
        
        feature_names = preprocessor.get_feature_names()
        importance_df = trainer.get_feature_importance(
            model_name='random_forest',
            feature_names=feature_names
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) > 0
    
    def test_cross_validation(self, trainer, training_data):
        """Test cross-validation scoring."""
        X, y, _ = training_data
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        cv_scores = trainer._cross_validate(model, X, y, cv=3)
        
        assert isinstance(cv_scores, np.ndarray)
        assert len(cv_scores) == 3
        assert all(score > 0 for score in cv_scores)  # RMSE should be positive


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create ModelEvaluator instance."""
        return ModelEvaluator()
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create trained model and test data."""
        preprocessor = DataPreprocessor()
        df = load_sample_data()
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        X, y = preprocessor.prepare_features(df_features, fit=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        trainer = ModelTrainer()
        model = trainer.train_single_model('linear_regression', X_train, y_train)
        
        return model, X_test, y_test, trainer.models
    
    def test_evaluate_model(self, evaluator, trained_model_and_data):
        """Test single model evaluation."""
        model, X_test, y_test, _ = trained_model_and_data
        
        metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            model_name='linear_regression'
        )
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
    
    def test_calculate_metrics(self, evaluator):
        """Test metrics calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        
        # Check values are reasonable
        assert metrics['rmse'] == pytest.approx(10.0, rel=0.1)
        assert metrics['mae'] == pytest.approx(10.0, rel=0.1)
        assert 0 <= metrics['r2'] <= 1
    
    def test_evaluate_all_models(self, evaluator, trained_model_and_data):
        """Test evaluation of multiple models."""
        _, X_test, y_test, models = trained_model_and_data
        
        comparison_df = evaluator.evaluate_all_models(models, X_test, y_test)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'model' in comparison_df.columns
        assert len(comparison_df) == len(models)
    
    def test_get_comparison_table(self, evaluator, trained_model_and_data):
        """Test comparison table generation."""
        model, X_test, y_test, _ = trained_model_and_data
        
        evaluator.evaluate_model(model, X_test, y_test, 'test_model')
        comparison_df = evaluator.get_comparison_table()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'model' in comparison_df.columns
        assert 'rmse' in comparison_df.columns
    
    def test_generate_report(self, evaluator, trained_model_and_data):
        """Test report generation."""
        model, X_test, y_test, _ = trained_model_and_data
        
        evaluator.evaluate_model(model, X_test, y_test, 'test_model')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'report.md'
            report = evaluator.generate_report(output_path=report_path)
            
            assert isinstance(report, str)
            assert len(report) > 0
            assert 'Model Evaluation Report' in report
            assert report_path.exists()
    
    def test_plot_predictions(self, evaluator, trained_model_and_data):
        """Test predictions plot generation."""
        model, X_test, y_test, _ = trained_model_and_data
        
        evaluator.evaluate_model(model, X_test, y_test, 'test_model')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'predictions.png'
            
            # This should not raise an error
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for testing
            
            evaluator.plot_predictions(
                model_name='test_model',
                save_path=plot_path
            )
            
            assert plot_path.exists()


class TestModelPredictor:
    """Tests for ModelPredictor class."""
    
    @pytest.fixture
    def saved_model_and_preprocessor(self):
        """Create and save model and preprocessor for testing."""
        preprocessor = DataPreprocessor()
        df = load_sample_data()
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        X, y = preprocessor.prepare_features(df_features, fit=True)
        
        # Train model
        trainer = ModelTrainer()
        trainer.train_single_model('random_forest', X, y, params={
            'n_estimators': 10,
            'random_state': 42
        })
        trainer.best_model = trainer.models['random_forest']
        trainer.best_model_name = 'random_forest'
        
        # Save to temp directory
        tmpdir = tempfile.mkdtemp()
        trainer.save_models(save_all=True, save_path=tmpdir)
        preprocessor.save_preprocessor(Path(tmpdir) / 'preprocessor.pkl')
        
        return tmpdir, df
    
    def test_load_model(self, saved_model_and_preprocessor):
        """Test model loading."""
        tmpdir, _ = saved_model_and_preprocessor
        
        predictor = ModelPredictor(
            model_path=Path(tmpdir) / 'best_model.pkl'
        )
        
        assert predictor.model is not None
    
    def test_predict_dataframe(self, saved_model_and_preprocessor):
        """Test prediction with DataFrame input."""
        tmpdir, df = saved_model_and_preprocessor
        
        predictor = ModelPredictor(
            model_path=Path(tmpdir) / 'best_model.pkl',
            preprocessor_path=Path(tmpdir) / 'preprocessor.pkl'
        )
        
        # Use first row for prediction
        test_data = df.iloc[[0]].drop('selling_price', axis=1)
        predictions = predictor.predict(test_data)
        
        assert predictions is not None
        assert len(predictions) == 1
        assert predictions[0] > 0
    
    def test_predict_dict(self, saved_model_and_preprocessor):
        """Test prediction with dictionary input."""
        tmpdir, _ = saved_model_and_preprocessor
        
        predictor = ModelPredictor(
            model_path=Path(tmpdir) / 'best_model.pkl',
            preprocessor_path=Path(tmpdir) / 'preprocessor.pkl'
        )
        
        test_data = {
            'year': 2018,
            'km_driven': 50000,
            'fuel': 'Petrol',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'First Owner',
            'mileage(km/ltr/kg)': '18.5 kmpl',
            'engine': '1200 CC',
            'max_power': '85 bhp',
            'seats': 5
        }
        
        predictions = predictor.predict(test_data)
        
        assert predictions is not None
        assert len(predictions) == 1
        assert predictions[0] > 0
    
    def test_predict_single(self, saved_model_and_preprocessor):
        """Test single prediction with individual parameters."""
        tmpdir, _ = saved_model_and_preprocessor
        
        predictor = ModelPredictor(
            model_path=Path(tmpdir) / 'best_model.pkl',
            preprocessor_path=Path(tmpdir) / 'preprocessor.pkl'
        )
        
        prediction = predictor.predict_single(
            year=2018,
            km_driven=50000,
            fuel='Petrol',
            seller_type='Individual',
            transmission='Manual',
            owner='First Owner',
            mileage=18.5,
            engine=1200,
            max_power=85,
            seats=5
        )
        
        assert prediction is not None
        assert isinstance(prediction, float)
        assert prediction > 0
    
    def test_predict_with_confidence(self, saved_model_and_preprocessor):
        """Test prediction with confidence intervals."""
        tmpdir, _ = saved_model_and_preprocessor
        
        predictor = ModelPredictor(
            model_path=Path(tmpdir) / 'best_model.pkl',
            preprocessor_path=Path(tmpdir) / 'preprocessor.pkl'
        )
        
        test_data = {
            'year': 2018,
            'km_driven': 50000,
            'fuel': 'Petrol',
            'seller_type': 'Individual',
            'transmission': 'Manual',
            'owner': 'First Owner',
            'mileage(km/ltr/kg)': '18.5 kmpl',
            'engine': '1200 CC',
            'max_power': '85 bhp',
            'seats': 5
        }
        
        result = predictor.predict_with_confidence(test_data)
        
        assert 'prediction' in result
        assert 'lower_95' in result
        assert 'upper_95' in result
    
    def test_batch_predict(self, saved_model_and_preprocessor):
        """Test batch prediction from file."""
        tmpdir, df = saved_model_and_preprocessor
        
        # Save test data to file
        test_file = Path(tmpdir) / 'test_data.csv'
        df.drop('selling_price', axis=1).to_csv(test_file, index=False)
        
        predictor = ModelPredictor(
            model_path=Path(tmpdir) / 'best_model.pkl',
            preprocessor_path=Path(tmpdir) / 'preprocessor.pkl'
        )
        
        output_file = Path(tmpdir) / 'predictions.csv'
        result_df = predictor.batch_predict(
            filepath=test_file,
            output_path=output_file
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'predicted_price' in result_df.columns
        assert len(result_df) == len(df)
        assert output_file.exists()


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete training and prediction pipeline."""
        # Load data
        df = load_sample_data()
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(df)
        df_features = preprocessor.create_features(df_clean)
        
        # Split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df_features, test_size=0.2, random_state=42
        )
        
        # Prepare features
        X_train, y_train = preprocessor.prepare_features(train_df, fit=True)
        X_test, y_test = preprocessor.prepare_features(test_df, fit=False)
        
        # Train
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        # Select best
        best_name, best_model = trainer.select_best_model(X_test, y_test)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(best_model, X_test, y_test, best_name)
        
        # Assertions
        assert best_model is not None
        assert metrics['rmse'] > 0
        assert 0 <= metrics['r2'] <= 1 or metrics['r2'] < 0  # R2 can be negative for bad fits
        
        # Save and load for prediction
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_models(save_path=tmpdir)
            preprocessor.save_preprocessor(Path(tmpdir) / 'preprocessor.pkl')
            
            # Load and predict
            predictor = ModelPredictor(
                model_path=Path(tmpdir) / 'best_model.pkl',
                preprocessor_path=Path(tmpdir) / 'preprocessor.pkl'
            )
            
            # Make prediction
            test_row = df.iloc[0].to_dict()
            del test_row['selling_price']
            
            prediction = predictor.predict(test_row)
            
            assert prediction[0] > 0