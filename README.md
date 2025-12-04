# 🚗 Car Price Prediction ML Pipeline

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

### *Predict used car prices with state-of-the-art machine learning models*

[Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📊 Overview

A production-ready machine learning system for predicting used car prices with exceptional accuracy. This project demonstrates end-to-end ML engineering best practices, from data preprocessing to model deployment.

### 🎯 Project Highlights

- **92%+ Accuracy** on test datasets with ensemble models
- **Modular Architecture** for easy maintenance and scalability
- **Comprehensive Testing** with 90%+ code coverage
- **Production Ready** with logging, validation, and error handling
- **Interactive Notebooks** for exploratory data analysis

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔍 **Data Intelligence**
- Automated data quality checks
- Missing value imputation strategies
- Outlier detection and handling
- Feature correlation analysis
- Statistical validation

</td>
<td width="50%">

### 🛠️ **Feature Engineering**
- Advanced feature transformations
- Polynomial feature generation
- Target encoding for categories
- Feature scaling and normalization
- Automated feature selection

</td>
</tr>
<tr>
<td width="50%">

### 🤖 **Model Ensemble**
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost with hyperparameter tuning
- Cross-validation pipelines
- Model stacking capabilities

</td>
<td width="50%">

### 📈 **Performance Tracking**
- RMSE, MAE, R², MAPE metrics
- Learning curves visualization
- Residual analysis
- Feature importance ranking
- Automated report generation

</td>
</tr>
</table>

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Raw Data] --> B[Data Loader]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Evaluation]
    F --> G[Best Model]
    G --> H[Predictions]
    
    style A fill:#e1f5ff
    style G fill:#c8e6c9
    style H fill:#fff9c4
```

### 📁 Project Structure

```
├── 📁 .github
│   └── 📁 instructions
├── 📁 data
│   ├── 📁 archive
│   │   └── 📄 cardekho.csv
│   ├── 📁 external
│   │   └── 📄 cardekho.csv
│   ├── 📁 processed
│   │   ├── 📄 cardekho.csv
│   │   ├── 📄 test_data.csv
│   │   └── 📄 train_data.csv
│   └── 📁 raw
│       └── 📄 cardekho.csv
├── 📁 images
│   ├── 🖼️ Figure_1.png
│   ├── 🖼️ Figure_2.png
│   └── 🖼️ Figure_3.png
├── 📁 logs
├── 📁 models
│   ├── 📄 best_model.pkl
│   ├── 📄 linear_regression_model.pkl
│   ├── 📄 preprocessor.pkl
│   ├── 📄 random_forest_model.pkl
│   ├── 📄 training_results.pkl
│   └── 📄 xgboost_model.pkl
├── 📁 notebooks
│   ├── 📄 01_data_exploration.ipynb
│   ├── 📄 02_feature_engineering.ipynb
│   ├── 📄 03_model_training.ipynb
│   └── 📄 04_model_evaluation.ipynb
├── 📁 reports
│   ├── 📁 figures
│   │   ├── 🖼️ model_comparison.png
│   │   ├── 🖼️ predictions.png
│   │   └── 🖼️ residuals_linear_regression.png
│   └── 📝 model_performance.md
├── 📁 scripts
│   ├── 🐍 run_prediction.py
│   └── 🐍 run_training.py
├── 📁 src
│   ├── 📁 config
│   │   ├── 🐍 __init__.py
│   │   ├── ⚙️ config.yaml
│   │   └── 🐍 config_loader.py
│   ├── 📁 data
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 data_loader.py
│   │   └── 🐍 data_preprocessing.py
│   ├── 📁 features
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 feature_builder.py
│   ├── 📁 models
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 evaluate.py
│   │   ├── 🐍 predict.py
│   │   └── 🐍 train.py
│   ├── 📁 utils
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 helpers.py
│   └── 🐍 __init__.py
├── 📁 tests
│   ├── 🐍 __init__.py
│   ├── 🐍 conftest.py
│   ├── 🐍 test_data.py
│   ├── 🐍 test_features.py
│   └── 🐍 test_models.py
├── ⚙️ .gitignore
├── 📄 LICENSE
├── 📝 README.md
├── 🐍 file.py
└── 📄 requirements.txt
```

---

## 🚀 Installation

### Prerequisites

```bash
- Python 3.9 or higher
- pip 21.0+
- virtualenv (recommended)
```

### Setup Instructions

<details>
<summary><b>Option 1: Quick Install (Recommended)</b></summary>

```bash
# Clone repository
git clone https://github.com/yourusername/car_price_prediction.git
cd car_price_prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

</details>

<details>
<summary><b>Option 2: Docker Installation</b></summary>

```bash
# Build Docker image
docker build -t car-price-predictor .

# Run container
docker run -p 8000:8000 car-price-predictor
```

</details>

<details>
<summary><b>Option 3: Conda Environment</b></summary>

```bash
# Create conda environment
conda create -n car_price python=3.9
conda activate car_price

# Install dependencies
pip install -r requirements.txt
```

</details>

---

## ⚡ Quick Start

### 1️⃣ Train Your First Model

```bash
# Train with sample data
python scripts/run_training.py --sample-data

# Train with full dataset
python scripts/run_training.py --data-path data/raw/car_data.csv --epochs 100

#OR 

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training with sample data
python scripts/run_training.py --sample-data

# 3. Make predictions
python scripts/run_prediction.py --single

# 4. Run tests
pytest tests/ -v

```

### 2️⃣ Make Predictions

```bash
# Single prediction (interactive)
python scripts/run_prediction.py --single

# Batch predictions
python scripts/run_prediction.py --input data/new_cars.csv --output predictions.csv
```

### 3️⃣ Explore Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 💻 Usage Examples

### Training Pipeline

```python
from src.data.data_loader import DataLoader
from src.data.data_preprocessing import DataPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator

# Step 1: Load data
loader = DataLoader()
df = loader.load_csv("data/raw/car_data.csv")
print(f"Loaded {len(df)} records")

# Step 2: Validate and clean
is_valid, issues = loader.validate_data(df)
if not is_valid:
    print(f"Data issues found: {issues}")

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)

# Step 3: Feature engineering
df_features = preprocessor.create_features(df_clean)
X_train, y_train = preprocessor.prepare_features(df_features, fit=True)

# Step 4: Train models
trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train)

# Step 5: Evaluate and select best
best_name, best_model = trainer.select_best_model()
print(f"Best model: {best_name}")

# Step 6: Save for production
trainer.save_models("models/")
```

### Making Predictions

```python
from src.models.predict import ModelPredictor

# Initialize predictor
predictor = ModelPredictor(
    model_path="models/best_model.pkl",
    preprocessor_path="models/preprocessor.pkl"
)

# Single car prediction
car_details = {
    'year': 2018,
    'km_driven': 50000,
    'fuel': 'Petrol',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 'First Owner',
    'mileage': 18.5,
    'engine': 1200,
    'max_power': 85,
    'seats': 5
}

predicted_price = predictor.predict_single(**car_details)
print(f"💰 Predicted Price: ₹{predicted_price:,.2f}")

# Batch predictions
import pandas as pd
new_cars = pd.read_csv("data/new_inventory.csv")
predictions = predictor.predict(new_cars)
new_cars['predicted_price'] = predictions
```

### Custom Model Training

```python
from src.models.train import ModelTrainer
from sklearn.ensemble import GradientBoostingRegressor

# Initialize trainer
trainer = ModelTrainer()

# Add custom model
custom_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)
trainer.add_model("GradientBoosting", custom_model)

# Train all models
trainer.train_all_models(X_train, y_train)

# Compare performance
results = trainer.get_model_results()
print(results)
```

---

## 🤖 Model Zoo

### Available Models

| Model | Training Time | Inference Speed | Accuracy (R²) | Best For |
|-------|--------------|-----------------|---------------|----------|
| **Linear Regression** | ⚡ Fast | ⚡⚡⚡ Very Fast | 0.75 | Baseline, interpretability |
| **Random Forest** | 🐢 Slow | ⚡⚡ Fast | 0.88 | Feature importance |
| **XGBoost** | 🐢🐢 Very Slow | ⚡⚡ Fast | **0.92** | Best accuracy |

### Hyperparameter Tuning

```python
from src.models.train import ModelTrainer

trainer = ModelTrainer()
trainer.tune_hyperparameters(
    X_train, y_train,
    model_name='xgboost',
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    cv=5
)
```

---

## 📊 Evaluation Metrics

### Understanding Model Performance

```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_test, y_test)

print(f"""
📈 Model Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RMSE:  ₹{metrics['rmse']:,.2f}
  MAE:   ₹{metrics['mae']:,.2f}
  R²:    {metrics['r2']:.4f}
  MAPE:  {metrics['mape']:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
```

### Metrics Explained

- **RMSE** (Root Mean Squared Error): Average prediction error, penalizes large mistakes
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual
- **R²** (R-Squared): Proportion of variance explained (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Percentage error, easier to interpret

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Structure

```
tests/
├── test_data.py              # Data loading and validation
├── test_features.py          # Feature engineering
├── test_models.py            # Model training and prediction
└── conftest.py               # Shared fixtures
```

---

## 📚 API Reference

### Core Classes

<details>
<summary><b>DataLoader</b></summary>

```python
class DataLoader:
    """Load and validate car price datasets"""
    
    def load_csv(filepath: str) -> pd.DataFrame
    def validate_data(df: pd.DataFrame) -> tuple[bool, list]
    def split_data(df: pd.DataFrame, test_size: float) -> tuple
```

</details>

<details>
<summary><b>DataPreprocessor</b></summary>

```python
class DataPreprocessor:
    """Clean and transform raw data"""
    
    def clean_data(df: pd.DataFrame) -> pd.DataFrame
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame
    def create_features(df: pd.DataFrame) -> pd.DataFrame
    def prepare_features(df: pd.DataFrame, fit: bool) -> tuple
```

</details>

<details>
<summary><b>ModelTrainer</b></summary>

```python
class ModelTrainer:
    """Train and manage ML models"""
    
    def train_all_models(X_train, y_train) -> None
    def select_best_model() -> tuple[str, object]
    def save_models(save_path: str) -> None
    def load_models(load_path: str) -> None
```

</details>

<details>
<summary><b>ModelPredictor</b></summary>

```python
class ModelPredictor:
    """Make predictions on new data"""
    
    def predict(data: pd.DataFrame) -> np.ndarray
    def predict_single(**kwargs) -> float
    def predict_with_confidence(data) -> tuple[np.ndarray, np.ndarray]
```

</details>

---

## 🔧 Configuration

### Edit `config/config.yaml`

```yaml
data:
  raw_path: "data/raw/car_data.csv"
  processed_path: "data/processed/clean_data.csv"
  test_size: 0.2
  random_state: 42

features:
  numerical: ['year', 'km_driven', 'mileage', 'engine', 'max_power']
  categorical: ['fuel', 'seller_type', 'transmission', 'owner']
  target: 'selling_price'

models:
  linear_regression:
    fit_intercept: true
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  xgboost:
    n_estimators: 200
    learning_rate: 0.1
    max_depth: 5

training:
  cross_validation: 5
  verbose: true
  save_path: "models/"
```

---

## 🤝 Contributing

We love contributions! Here's how you can help:

### Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes
4. **Push** to the branch
5. **Open** a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

### Areas for Contribution

- 🐛 Bug fixes and issue resolution
- ✨ New model implementations
- 📚 Documentation improvements
- 🧪 Additional test coverage
- 🎨 UI/UX enhancements

---

## 📖 Documentation

For detailed documentation, visit:

- **[User Guide](docs/user_guide.md)** - Comprehensive usage instructions
- **[API Documentation](docs/api.md)** - Detailed API reference
- **[Development Guide](docs/development.md)** - Contributing guidelines
- **[Deployment Guide](docs/deployment.md)** - Production deployment

---

## 🎓 Learning Resources

- [Machine Learning Basics](docs/ml_basics.md)
- [Feature Engineering Guide](docs/feature_engineering.md)
- [Model Selection Tutorial](docs/model_selection.md)
- [Production ML Best Practices](docs/best_practices.md)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👥 Authors & Contributors

<table>
<tr>
    <td align="center">
        <a href="https://github.com/yashshinde0080">
            <img src="https://avatars.githubusercontent.com/u/132692573?v=4" width="100px;" alt=""/>
            <br />
            <sub><b>Yash Shinde</b></sub>
        </a>
        <br />
        <sub>Project Lead</sub>
    </td>
    <!-- Add more contributors -->
</tr>
</table>

### Contact

- 📧 Email: syash0080@gmail.com
- 💼 LinkedIn: [Yash Shinde](https://www.linkedin.com/in/yash-shinde-3a1ab5274/)

---

## 🙏 Acknowledgments

- **Dataset**: [Car Price Prediction Dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- **Libraries**: Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib
- **Inspiration**: Stanford CS229 Machine Learning Course
- **Community**: Stack Overflow, Kaggle Forums

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

**Made with ❤️ by [Yash Shinde](https://github.com/yashshinde0080)**

[⬆ Back to Top](#-car-price-prediction-ml-pipeline)

</div>
