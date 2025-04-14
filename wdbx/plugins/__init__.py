"""
Plugin system for WDBX.
"""

import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import WDBXPlugin, PluginError, PluginManager

logger = logging.getLogger(__name__)


def load_plugins(wdbx) -> Dict[str, WDBXPlugin]:
    """
    Load and initialize plugins.

    Args:
        wdbx: The WDBX instance

    Returns:
        Dictionary mapping plugin names to plugin instances
    """
    # Create plugin manager
    manager = PluginManager(wdbx)

    # Get plugin directory from configuration
    plugin_dir = wdbx.config.get("WDBX_PLUGIN_DIRECTORY")
    auto_discover = wdbx.config.get("WDBX_PLUGIN_AUTO_DISCOVER", True)

    # Load plugins
    plugins = manager.load_plugins(plugin_dir, auto_discover)

    return plugins


# Export plugin classes
__all__ = ["WDBXPlugin", "PluginError", "PluginManager", "load_plugins"]
