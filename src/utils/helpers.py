"""
Utility functions and helper methods for the car price prediction project.
"""

import os
import sys
import pickle
import joblib
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root directory.
    """
    current_path = Path(__file__).resolve()
    
    # Navigate up to find project root (where src folder is)
    for parent in current_path.parents:
        if (parent / "src").is_dir() and (parent / "requirements.txt").is_file():
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


def setup_logging(
    log_level: str = "INFO",
    log_path: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup logging configuration using loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_path: Path to log file. If None, only console logging.
        log_format: Custom log format string.
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if path provided
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format=log_format,
            level=log_level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
        logger.info(f"Logging to file: {log_path}")


def create_directories(directories: List[Union[str, Path]]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create.
    """
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")


def save_pickle(obj: Any, filepath: Union[str, Path], use_joblib: bool = True) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save.
        filepath: Path to save the pickle file.
        use_joblib: Use joblib for saving (better for large numpy arrays).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if use_joblib:
        joblib.dump(obj, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    logger.info(f"Object saved to {filepath}")


def load_pickle(filepath: Union[str, Path], use_joblib: bool = True) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to the pickle file.
        use_joblib: Use joblib for loading.
        
    Returns:
        Loaded object.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if use_joblib:
        obj = joblib.load(filepath)
    else:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    
    logger.info(f"Object loaded from {filepath}")
    return obj


def format_number(number: float, precision: int = 2) -> str:
    """
    Format a number with comma separators and specified precision.
    
    Args:
        number: Number to format.
        precision: Decimal precision.
        
    Returns:
        Formatted string.
    """
    return f"{number:,.{precision}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value.
        new_value: New value.
        
    Returns:
        Percentage change.
    """
    if old_value == 0:
        return float('inf') if new_value != 0 else 0.0
    return ((new_value - old_value) / abs(old_value)) * 100


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of a DataFrame.
    
    Args:
        df: Pandas DataFrame.
        
    Returns:
        Human-readable memory usage string.
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    if memory_bytes < 1024:
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024 ** 2:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024 ** 3:
        return f"{memory_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{memory_bytes / (1024 ** 3):.2f} GB"


def timer_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function.
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")