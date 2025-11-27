#!/usr/bin/env python3
"""
create_ml_project.py
Automatically creates a professional, best-practice Machine Learning project structure.
"""

import os
from pathlib import Path
import json
import argparse
import textwrap

def _write_file(path: Path, content: str, overwrite: bool = False, binary: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    mode = "wb" if binary else "w"
    data = content.encode("utf-8") if binary else content
    with open(path, mode) as f:
        f.write(data)
    return True

def _create_notebook(path: Path):
    nb = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    _write_file(path, json.dumps(nb, indent=2))

def create_project_structure(project_name: str = "my_ml_project"):
    base_path = Path(project_name)
    base_path.mkdir(exist_ok=True)

    # Files and directories to create (mapping relative path -> content)
    files = {
        "README.md": textwrap.dedent(f"""\
            # {project_name.replace('_', ' ').title()}

            A clean, scalable, and professional machine learning project template.

            ## Project Structure

            - data/: raw, processed, external datasets
            - notebooks/: exploratory and model notebooks
            - src/: source code (data, features, models, config)
            - models/: saved models
            - tests/: unit tests
            """),
        "requirements.txt": textwrap.dedent("""\
            pandas
            numpy
            scikit-learn
            jupyter
            pyyaml
            matplotlib
            seaborn
            tqdm
            joblib
            """),
        ".gitignore": textwrap.dedent("""\
            __pycache__/
            *.py[cod]
            .env
            .venv/
            venv/
            env/
            .ipynb_checkpoints/
            .DS_Store
            models/*.pkl
            """),
        "LICENSE": textwrap.dedent("""\
            MIT License

            Copyright (c) YEAR

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction...
            """),
        # src config file
        "src/config/config.yaml": textwrap.dedent("""\
            # Configuration file
            data:
              raw_path: ../data/raw
              processed_path: ../data/processed
              external_path: ../data/external

            model:
              name: best_model
              save_path: ../models

            training:
              test_size: 0.2
              random_state: 42
              n_estimators: 100
              max_depth: null
            """),
        # src __init__.py
        "src/__init__.py": '"""Top-level package for project."""\n',
        # data scripts
        "src/data/data_loader.py": textwrap.dedent('''\
            """Data loading utilities."""
            import pandas as pd
            from pathlib import Path

            def load_data(path: str) -> pd.DataFrame:
                """Load dataset from CSV file."""
                return pd.read_csv(path)
            '''),
        "src/data/data_preprocessing.py": textwrap.dedent('''\
            """Data preprocessing and cleaning functions."""
            import pandas as pd

            def clean_data(df: pd.DataFrame) -> pd.DataFrame:
                """Apply basic cleaning steps to the dataset."""
                df = df.copy()
                df.drop_duplicates(inplace=True)
                # Add more cleaning steps as needed
                return df

            def split_features_target(df: pd.DataFrame, target: str):
                """Separate features and target variable."""
                X = df.drop(columns=[target])
                y = df[target]
                return X, y
            '''),
        # features
        "src/features/feature_builder.py": textwrap.dedent('''\
            """Feature engineering functions."""
            import pandas as pd

            def build_features(df: pd.DataFrame) -> pd.DataFrame:
                """Create new features from raw data."""
                df = df.copy()
                # Example: create interaction features, encode categoricals, etc.
                return df
            '''),
        # models scripts
        "src/models/train.py": textwrap.dedent('''\
            """Model training script."""
            import joblib
            import yaml
            from pathlib import Path
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Load config
            with open("src/config/config.yaml", "r") as f:
                config = yaml.safe_load(f)

            def train_model(X, y, config):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=config['training']['test_size'],
                    random_state=config['training']['random_state']
                )

                model = RandomForestClassifier(
                    n_estimators=config['training']['n_estimators'],
                    max_depth=config['training'].get('max_depth'),
                    random_state=config['training']['random_state']
                )
                model.fit(X_train, y_train)

                # Save model
                model_path = Path(config['model']['save_path']) / f"{config['model']['name']}.pkl"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                print(f"Model saved to {model_path}")

                return model, X_test, y_test

            if __name__ == "__main__":
                print("Training pipeline placeholder")
            '''),
        "src/models/evaluate.py": textwrap.dedent('''\
            """Model evaluation script."""
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            import joblib

            def evaluate_model(model, X_test, y_test):
                predictions = model.predict(X_test)
                print("Accuracy:", accuracy_score(y_test, predictions))
                print("\\nClassification Report:")
                print(classification_report(y_test, predictions))
                print("\\nConfusion Matrix:")
                print(confusion_matrix(y_test, predictions))

            if __name__ == "__main__":
                # Example usage
                # model = joblib.load("../models/best_model.pkl")
                # evaluate_model(model, X_test, y_test)
                pass
            '''),
        "src/models/predict.py": textwrap.dedent('''\
            """Prediction script for new data."""
            import joblib
            import pandas as pd

            def load_model(model_path: str = "../models/best_model.pkl"):
                return joblib.load(model_path)

            def predict_new_data(model, data: pd.DataFrame):
                """Make predictions on new data."""
                return model.predict(data)

            def predict_single(sample: dict):
                model = load_model()
                df = pd.DataFrame([sample])
                return predict_new_data(model, df)[0]

            if __name__ == "__main__":
                print("Prediction script ready")
            '''),
        # tests
        "tests/test_data.py": textwrap.dedent('''\
            import pytest
            from src.data.data_loader import load_data

            def test_load_data():
                # Add real tests when data exists
                pass
            '''),
        "tests/test_features.py": textwrap.dedent('''\
            import pytest
            from src.features.feature_builder import build_features
            import pandas as pd

            def test_build_features():
                df = pd.DataFrame({"A": [1, 2, 3]})
                result = build_features(df)
                assert isinstance(result, pd.DataFrame)
            '''),
        "tests/test_models.py": textwrap.dedent('''\
            import pytest
            from src.models.train import train_model

            def test_train_model():
                # Minimal test - will expand with fixtures
                pass
            '''),
    }

    # Directories to ensure exist
    dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src",
        "src/config",
        "src/data",
        "src/features",
        "src/models",
        "models",
        "tests"
    ]

    created = {"dirs": [], "files": []}

    # Create directories
    for d in dirs:
        p = base_path / d
        p.mkdir(parents=True, exist_ok=True)
        created["dirs"].append(str(p))

    # Create files
    for rel, content in files.items():
        p = base_path / rel
        ok = _write_file(p, content)
        if ok:
            created["files"].append(str(p))

    # Create notebooks (minimal)
    notebooks = [
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_model_training.ipynb",
        "notebooks/04_model_evaluation.ipynb",
    ]
    for nb in notebooks:
        p = base_path / nb
        # Only write if not exists
        if not p.exists():
            _create_notebook(p)
            created["files"].append(str(p))

    # Create a placeholder model file (binary empty file)
    model_placeholder = base_path / "models" / "best_model.pkl"
    if not model_placeholder.exists():
        _write_file(model_placeholder, "", binary=True)
        created["files"].append(str(model_placeholder))

    # Summary print
    print(f"Project '{project_name}' created at: {base_path.resolve()}")
    print(f"Directories created: {len(created['dirs'])}")
    print(f"Files created: {len(created['files'])}")
    return created

def main():
    parser = argparse.ArgumentParser(description="Create ML project structure.")
    parser.add_argument("name", nargs="?", default="project_name", help="Name of the project folder to create")
    args = parser.parse_args()
    create_project_structure(args.name)

if __name__ == "__main__":
    main()