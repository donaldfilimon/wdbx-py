"""
Base plugin class for WDBX.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Exception raised when a plugin operation fails."""
    pass


class WDBXPlugin(ABC):
    """
    Base class for WDBX plugins.

    All plugins must inherit from this class and implement its abstract methods.

    Attributes:
        wdbx: Reference to the WDBX instance that loaded the plugin
    """

    def __init__(self, wdbx):
        """
        Initialize the plugin.

        Args:
            wdbx: Reference to the WDBX instance
        """
        self.wdbx = wdbx
        self.logger = logging.getLogger(f"wdbx.plugins.{self.name}")
        self.logger.debug(f"Initializing plugin: {self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the plugin.

        Returns:
            Plugin name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the description of the plugin.

        Returns:
            Plugin description
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Get the version of the plugin.

        Returns:
            Plugin version
        """
        pass

    async def initialize(self) -> None:
        """
        Perform asynchronous initialization.

        This method is called after the plugin is loaded.
        It should be used to initialize resources, connect to external services, etc.
        """
        pass

    async def shutdown(self) -> None:
        """
        Clean up resources when shutting down.

        This method is called when the WDBX instance is shutting down.
        It should be used to close connections, release resources, etc.
        """
        pass

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats

        Raises:
            PluginError: If embedding creation fails
        """
        raise PluginError(
            f"Plugin {self.name} does not implement create_embedding")

    def register_commands(self) -> None:
        """
        Register commands with the WDBX CLI.

        This method is called after the plugin is loaded, if the CLI is available.
        It should be used to register CLI commands provided by the plugin.
        """
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from the WDBX instance.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        # Check for plugin-specific configuration
        plugin_key = f"WDBX_{self.name.upper()}_{key}"
        if self.wdbx.config.has(plugin_key):
            return self.wdbx.config.get(plugin_key)

        # Use general configuration
        wdbx_key = f"WDBX_{key}"
        return self.wdbx.config.get(wdbx_key, default)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the plugin.

        Returns:
            Dictionary with statistics
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }

    def get_help(self) -> str:
        """
        Get help information about the plugin.

        Returns:
            Help information as a string
        """
        return f"""
Plugin: {self.name} (v{self.version})
{self.description}

Configuration:
    This plugin supports the following configuration options:
    (Set these as environment variables or in the configuration dictionary)

    WDBX_{self.name.upper()}_*: Plugin-specific configuration options

Methods:
    initialize(): Perform asynchronous initialization
    shutdown(): Clean up resources when shutting down
    get_config(key, default=None): Get a configuration value
    get_stats(): Get statistics about the plugin
    get_help(): Get help information about the plugin

Additional plugin-specific methods may be available.
See the plugin documentation for details.
"""


class PluginManager:
    """
    Manager for WDBX plugins.

    This class handles loading, initializing, and accessing plugins.

    Attributes:
        wdbx: Reference to the WDBX instance
        plugins: Dictionary of loaded plugins
    """

    def __init__(self, wdbx):
        """
        Initialize the plugin manager.

        Args:
            wdbx: Reference to the WDBX instance
        """
        self.wdbx = wdbx
        self.plugins = {}
        self.logger = logging.getLogger("wdbx.plugins")

    def load_plugins(self, plugin_dir: Optional[str] = None,
                     auto_discover: bool = True) -> Dict[str, WDBXPlugin]:
        """
        Load plugins from the plugin directory.

        Args:
            plugin_dir: Directory containing plugins (default: built-in plugins)
            auto_discover: Whether to automatically discover plugins

        Returns:
            Dictionary of loaded plugins
        """
        from pathlib import Path
        import os
        import importlib
        import inspect

        # Default to built-in plugins
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent
        else:
            plugin_dir = Path(plugin_dir)

        self.logger.debug(f"Loading plugins from {plugin_dir}")

        # Load built-in plugins
        for file_path in plugin_dir.glob("*.py"):
            # Skip __init__.py, base.py, and other non-plugin files
            if file_path.name in [
                    "__init__.py", "base.py"] or file_path.name.startswith("_"):
                continue

            try:
                # Import module
                module_name = file_path.stem
                self.logger.debug(f"Attempting to load plugin: {module_name}")

                # Skip if not a proper plugin module
                if not module_name:
                    continue

                # Import module
                module_path = f"wdbx.plugins.{module_name}"

                try:
                    # Try importing as a built-in module
                    module = importlib.import_module(module_path)
                except ImportError:
                    # Try importing as a relative module
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                # Find plugin classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)

                    # Check if it's a plugin class
                    if (
                        inspect.isclass(attr) and
                        issubclass(attr, WDBXPlugin) and
                        attr is not WDBXPlugin
                    ):
                        # Create plugin instance
                        plugin = attr(self.wdbx)
                        plugin_name = plugin.name

                        # Register plugin
                        self.plugins[plugin_name] = plugin
                        self.logger.info(
                            f"Loaded plugin: {plugin_name} (v{plugin.version})")
                        break
                else:
                    self.logger.warning(
                        f"No plugin class found in {module_path}")
            except Exception as e:
                self.logger.error(
                    f"Error loading plugin from {file_path}: {e}")

        # Load external plugins if auto-discover is enabled
        if auto_discover:
            try:
                # Check for plugins in PYTHONPATH
                import pkg_resources
                for entry_point in pkg_resources.iter_entry_points(
                        "wdbx.plugins"):
                    try:
                        plugin_class = entry_point.load()
                        plugin = plugin_class(self.wdbx)
                        plugin_name = plugin.name

                        # Register plugin
                        self.plugins[plugin_name] = plugin
                        self.logger.info(
                            f"Loaded external plugin: {plugin_name} (v{plugin.version})")
                    except Exception as e:
                        self.logger.error(
                            f"Error loading external plugin {entry_point.name}: {e}")
            except Exception as e:
                self.logger.error(f"Error loading external plugins: {e}")

        return self.plugins

    async def initialize_plugins(self) -> None:
        """
        Initialize all loaded plugins.

        This method should be called after loading plugins.
        """
        import asyncio

        self.logger.debug(f"Initializing {len(self.plugins)} plugins")

        # Initialize plugins concurrently
        init_tasks = []
        for plugin_name, plugin in self.plugins.items():
            try:
                init_tasks.append(plugin.initialize())
            except Exception as e:
                self.logger.error(
                    f"Error initializing plugin {plugin_name}: {e}")

        # Wait for all plugins to initialize
        if init_tasks:
            await asyncio.gather(*init_tasks)

    async def shutdown_plugins(self) -> None:
        """
        Shut down all loaded plugins.

        This method should be called when the WDBX instance is shutting down.
        """
        import asyncio

        self.logger.debug(f"Shutting down {len(self.plugins)} plugins")

        # Shut down plugins concurrently
        shutdown_tasks = []
        for plugin_name, plugin in self.plugins.items():
            try:
                shutdown_tasks.append(plugin.shutdown())
            except Exception as e:
                self.logger.error(
                    f"Error shutting down plugin {plugin_name}: {e}")

        # Wait for all plugins to shut down
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks)

    def get_plugin(self, plugin_name: str) -> Optional[WDBXPlugin]:
        """
        Get a plugin by name.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance if found, None otherwise
        """
        return self.plugins.get(plugin_name)

    def register_plugin(self, plugin: WDBXPlugin) -> bool:
        """
        Register a plugin with the manager.

        Args:
            plugin: Plugin instance to register

        Returns:
            True if the plugin was registered, False if it already exists
        """
        plugin_name = plugin.name

        if plugin_name in self.plugins:
            self.logger.warning(f"Plugin {plugin_name} already registered")
            return False

        self.plugins[plugin_name] = plugin
        self.logger.info(
            f"Registered plugin: {plugin_name} (v{plugin.version})")
        return True

    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin from the manager.

        Args:
            plugin_name: Name of the plugin to unregister

        Returns:
            True if the plugin was unregistered, False if it wasn't found
        """
        if plugin_name not in self.plugins:
            self.logger.warning(f"Plugin {plugin_name} not registered")
            return False

        del self.plugins[plugin_name]
        self.logger.info(f"Unregistered plugin: {plugin_name}")
        return True

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all loaded plugins.

        Returns:
            List of dictionaries with plugin information
        """
        return [
            {
                "name": plugin.name,
                "description": plugin.description,
                "version": plugin.version,
            }
            for plugin in self.plugins.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the plugin manager.

        Returns:
            Dictionary with statistics
        """
        return {
            "num_plugins": len(self.plugins),
            "plugins": self.get_plugin_info(),
        }
