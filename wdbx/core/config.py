"""
Configuration management for WDBX.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)


class WDBXConfig:
    """
    Configuration manager for WDBX.

    This class handles configuration options from environment variables,
    configuration files, and runtime parameters.

    Attributes:
        config_dict (Dict[str, Any]): Dictionary containing configuration values
        config_sources (Dict[str, str]): Dictionary mapping config keys to their sources
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        # Core settings
        "VECTOR_STORE_SAVE_IMMEDIATELY": False,
        "VECTOR_STORE_THREADS": os.cpu_count() or 4,
        "VECTOR_STORE_CACHE_SIZE_MB": 128,

        # Index settings
        "HNSW_M": 16,
        "HNSW_EF_CONSTRUCTION": 200,
        "HNSW_EF_SEARCH": 50,
        "FAISS_INDEX_TYPE": "Flat",
        "FAISS_NPROBE": 8,

        # Distributed settings
        "DISTRIBUTED_HOST": "localhost",
        "DISTRIBUTED_PORT": 7777,
        "DISTRIBUTED_AUTH_ENABLED": False,
        "DISTRIBUTED_AUTH_KEY": "",

        # Plugin settings
        "PLUGIN_DIRECTORY": "plugins",
        "PLUGIN_AUTO_DISCOVER": True,
        "PLUGIN_TIMEOUT": 30,
    }

    def __init__(
            self, config_dict: Optional[Dict[str, Any]] = None,
            config_path: Optional[str] = None):
        """
        Initialize a new configuration manager.

        Args:
            config_dict: Optional dictionary of configuration values
            config_path: Optional path to configuration file
        """
        self.config_dict = {}
        self.config_sources = {}

        # Load default configuration
        self.config_dict.update(self.DEFAULT_CONFIG)
        for key in self.DEFAULT_CONFIG:
            self.config_sources[key] = "default"

        # Load configuration from file if specified
        if config_path:
            self._load_config_from_file(config_path)

        # Load configuration from environment variables
        self._load_config_from_env()

        # Apply provided configuration (highest precedence)
        if config_dict:
            self.config_dict.update(config_dict)
            for key in config_dict:
                self.config_sources[key] = "runtime"

        logger.debug(f"Initialized config with {len(self.config_dict)} values")

    def _load_config_from_file(self, config_path: str):
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return

        try:
            with open(path, 'r') as f:
                file_config = json.load(f)

            # Update configuration
            self.config_dict.update(file_config)
            for key in file_config:
                self.config_sources[key] = f"file:{config_path}"

            logger.debug(f"Loaded configuration from file: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Malformed configuration file {config_path}: {e}")
        except Exception as e:
            logger.error(
                f"Error loading configuration from file {config_path}: {e}")

    def _load_config_from_env(self):
        """Load configuration from environment variables."""
        # Look for any environment variables starting with "WDBX_"
        for key, value in os.environ.items():
            if key.startswith("WDBX_"):
                # Convert environment variable to configuration key
                # We keep the "WDBX_" prefix for consistency
                config_key = key

                # Try to convert value to appropriate type
                typed_value = self._parse_value(value)

                # Update configuration
                self.config_dict[config_key] = typed_value
                self.config_sources[config_key] = "environment"

        logger.debug(f"Loaded configuration from environment variables")

    def _parse_value(self, value: str) -> Any:
        """
        Parse a string value into an appropriate Python type.

        Args:
            value: String value to parse

        Returns:
            Parsed value in appropriate type
        """
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try other common types
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False
        elif value.isdigit():
            return int(value)
        elif value.replace(".", "", 1).isdigit() and value.count(".") <= 1:
            return float(value)

        # Default to string
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        return self.config_dict.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_dict[key] = value
        self.config_sources[key] = "runtime"

    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key: Configuration key

        Returns:
            True if the key exists, False otherwise
        """
        return key in self.config_dict

    def get_source(self, key: str) -> Optional[str]:
        """
        Get the source of a configuration value.

        Args:
            key: Configuration key

        Returns:
            Source of the configuration value, or None if key not found
        """
        return self.config_sources.get(key)

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dictionary of all configuration values
        """
        return self.config_dict.copy()

    def get_typed(
            self, key: str, expected_type: type, default: Any = None) -> Any:
        """
        Get a configuration value with type checking.

        Args:
            key: Configuration key
            expected_type: Expected type of the value
            default: Default value if key is not found or type doesn't match

        Returns:
            Configuration value or default
        """
        value = self.get(key, default)

        # If value is None or already the correct type, return it
        if value is None or isinstance(value, expected_type):
            return value

        # Try to convert to the expected type
        try:
            if expected_type is bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "on")
                return bool(value)
            elif expected_type is int:
                return int(value)
            elif expected_type is float:
                return float(value)
            elif expected_type is str:
                return str(value)
            elif expected_type is list:
                if isinstance(value, str):
                    return json.loads(value) if value.startswith("[") else value.split(",")
                return list(value)
            elif expected_type is dict:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
        except (ValueError, TypeError, json.JSONDecodeError):
            logger.warning(
                f"Could not convert config value {key}={value} to {expected_type.__name__}, using default")
            return default

        # If we can't convert, return default
        logger.warning(
            f"Could not convert config value {key}={value} to {expected_type.__name__}, using default")
        return default

    def save_to_file(self, config_path: str) -> bool:
        """
        Save the current configuration to a file.

        Args:
            config_path: Path to save the configuration file

        Returns:
            True if the configuration was saved successfully, False otherwise
        """
        try:
            path = Path(config_path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True)

            with open(path, 'w') as f:
                json.dump(self.config_dict, f, indent=2, sort_keys=True)

            logger.debug(f"Saved configuration to file: {config_path}")
            return True
        except Exception as e:
            logger.error(
                f"Error saving configuration to file {config_path}: {e}")
            return False

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self.config_dict = self.DEFAULT_CONFIG.copy()
        self.config_sources = {key: "default" for key in self.DEFAULT_CONFIG}

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to configuration values."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like setting of configuration values."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator to check if a key exists."""
        return self.has(key)

    def __len__(self) -> int:
        """Return the number of configuration values."""
        return len(self.config_dict)

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return f"WDBXConfig({self.config_dict})"
