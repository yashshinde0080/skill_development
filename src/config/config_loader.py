"""
Configuration loader module for loading and managing project configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class ConfigLoader:
    """
    Configuration loader class for loading YAML configuration files.
    
    Attributes:
        config_path (Path): Path to the configuration file.
        config (Dict): Loaded configuration dictionary.
    """
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        if self._config is None:
            if config_path is None:
                # Default config path
                config_path = Path(__file__).parent / "config.yaml"
            else:
                config_path = Path(config_path)
            
            self.config_path = config_path
            self._config = self._load_config()
            logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration parameters.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            yaml.YAMLError: If configuration file is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., 'data.raw_path').
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.
        
        Returns:
            Complete configuration dictionary.
        """
        return self._config.copy()
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            key: Dot-separated key path.
            value: New value to set.
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get or create configuration loader instance.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        ConfigLoader instance.
    """
    return ConfigLoader(config_path)