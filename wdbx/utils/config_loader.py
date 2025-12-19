"""
Configuration loader utility for WDBX.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(
    config_path: Optional[str] = None, env_prefix: str = "WDBX_"
) -> Dict[str, Any]:
    """
    Load WDBX configuration from file and environment variables.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        env_prefix: Prefix for environment variables

    Returns:
        Configuration dictionary
    """
    config = {}

    # Load from file if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            try:
                if config_file.suffix.lower() in [".yaml", ".yml"]:
                    with open(config_file, "r") as f:
                        file_config = yaml.safe_load(f)
                elif config_file.suffix.lower() == ".json":
                    with open(config_file, "r") as f:
                        file_config = json.load(f)
                else:
                    logger.warning(
                        f"Unsupported config file format: {config_file.suffix}"
                    )
                    file_config = {}

                # Convert keys to uppercase with prefix for consistency
                _update_config_recursive(config, file_config, env_prefix)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")

    # Load from environment variables
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Convert value to appropriate type
            config[key] = _parse_value(value)

    return config


def _update_config_recursive(
    config: Dict[str, Any], source: Dict[str, Any], prefix: str, current_path: str = ""
) -> None:
    """
    Recursively update configuration dictionary from source.

    Args:
        config: Target configuration dictionary
        source: Source configuration dictionary
        prefix: Environment variable prefix
        current_path: Current path in the configuration hierarchy
    """
    for key, value in source.items():
        # Build the full key path
        full_key = f"{current_path}_{key}" if current_path else key
        env_key = f"{prefix}{full_key.upper()}"

        if isinstance(value, dict):
            # Recurse into nested dictionaries
            _update_config_recursive(config, value, prefix, full_key)
        else:
            # Set value for leaf node
            config[env_key] = value


def _parse_value(value: str) -> Any:
    """
    Parse a string value into an appropriate Python type.

    Args:
        value: String value to parse

    Returns:
        Parsed value in appropriate type
    """
    # Try to parse as JSON first (for lists, dictionaries, booleans, null)
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


def save_config(config: Dict[str, Any], config_path: str, format: str = "yaml") -> bool:
    """
    Save configuration to a file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration to
        format: Format to save in ('yaml' or 'json')

    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = Path(config_path)

        # Create parent directories if they don't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to a hierarchical dictionary
        hierarchical_config = {}
        for key, value in config.items():
            if key.startswith("WDBX_"):
                # Remove prefix and convert to lowercase
                key = key[5:].lower()

            # Split by underscore to create hierarchy
            parts = key.split("_")
            current = hierarchical_config

            # Navigate through the hierarchy
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Set the value at the leaf node
                    current[part] = value
                else:
                    # Create intermediate dictionaries
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        # Save in requested format
        if format.lower() == "yaml":
            with open(config_file, "w") as f:
                yaml.dump(hierarchical_config, f, default_flow_style=False)
        elif format.lower() == "json":
            with open(config_file, "w") as f:
                json.dump(hierarchical_config, f, indent=2)
        else:
            logger.error(f"Unsupported config format: {format}")
            return False

        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        return False
