"""
Main WDBX class that provides the primary interface for the vector database.
"""

import os
import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

from .vector_store import VectorStore
from .config import WDBXConfig
from .distributed import ShardManager
from ..plugins.base import WDBXPlugin

logger = logging.getLogger(__name__)


class WDBX:
    """
    WDBX: Main class for the vector database system.

    This class provides the main interface for working with the WDBX vector database,
    including storing and searching vectors, managing plugins, and handling configuration.

    Attributes:
        vector_dim (int): Dimension of vectors stored in the database
        num_shards (int): Number of shards for distributed storage
        data_dir (str): Directory for storing vector data
        config (WDBXConfig): Configuration object
        plugins (Dict[str, WDBXPlugin]): Dictionary of loaded plugins
    """

    def __init__(
        self,
        vector_dimension: int = 384,
        num_shards: int = 1,
        data_dir: str = "./wdbx_data",
        config: Optional[Dict[str, Any]] = None,
        enable_plugins: bool = True,
        enable_distributed: bool = False,
        enable_gpu: bool = False,
        log_level: str = "INFO",
    ):
        """
        Initialize a new WDBX instance.

        Args:
            vector_dimension: Dimension of vectors stored in the database
            num_shards: Number of shards for distributed storage
            data_dir: Directory for storing vector data
            config: Configuration dictionary
            enable_plugins: Whether to enable the plugin system
            enable_distributed: Whether to enable distributed architecture
            enable_gpu: Whether to enable GPU acceleration (if available)
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        # Set up logging
        self._setup_logging(log_level)
        logger.info(f"Initializing WDBX v{self.version}")

        # Set core attributes
        self.vector_dim = vector_dimension
        self.num_shards = num_shards
        self.data_dir = Path(data_dir)
        self.config = WDBXConfig(config or {})
        self.enable_plugins = enable_plugins
        self.enable_distributed = enable_distributed
        self.enable_gpu = enable_gpu

        # Create data directory if it doesn't exist
        if not self.data_dir.exists():
            logger.info(f"Creating data directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True)

        # Initialize vector storage
        self._init_vector_store()

        # Initialize plugin system
        self.plugins = {}
        if self.enable_plugins:
            self._init_plugins()

        # Initialize distributed architecture if enabled
        self.shard_manager = None
        if self.enable_distributed:
            self._init_distributed()

        logger.info(
            f"WDBX initialized successfully with vector_dim={self.vector_dim}, num_shards={self.num_shards}"
        )

    @property
    def version(self) -> str:
        """Return the current WDBX version."""
        from .. import __version__

        return __version__

    def _setup_logging(self, log_level: str):
        """Set up logging for WDBX."""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")

        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=numeric_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    def _init_vector_store(self):
        """Initialize the vector storage system."""
        logger.debug("Initializing vector storage")

        self.vector_store = VectorStore(
            vector_dim=self.vector_dim,
            data_dir=self.data_dir,
            num_shards=self.num_shards,
            use_gpu=self.enable_gpu,
            config=self.config,
        )

    def _init_plugins(self):
        """Initialize the plugin system and load available plugins."""
        logger.debug("Initializing plugin system")

        # Import here to avoid circular imports
        from ..plugins import load_plugins

        # Load available plugins
        available_plugins = load_plugins(self)
        for plugin_name, plugin in available_plugins.items():
            self.plugins[plugin_name] = plugin
            logger.info(f"Loaded plugin: {plugin_name} v{plugin.version}")

    def _init_distributed(self):
        """Initialize the distributed architecture."""
        logger.debug("Initializing distributed architecture")

        self.shard_manager = ShardManager(
            num_shards=self.num_shards,
            data_dir=self.data_dir,
            config=self.config,
        )

    async def initialize(self):
        """
        Asynchronously initialize WDBX components.

        This method should be called after creating a WDBX instance to ensure
        all async components are properly initialized.
        """
        logger.debug("Performing async initialization")

        # Initialize vector store
        await self.vector_store.initialize()

        # Initialize plugins
        if self.enable_plugins:
            init_tasks = [plugin.initialize() for plugin in self.plugins.values()]
            if init_tasks:
                await asyncio.gather(*init_tasks)

        # Initialize distributed components
        if self.enable_distributed and self.shard_manager:
            await self.shard_manager.initialize()

        logger.info("WDBX async initialization complete")

    async def shutdown(self):
        """
        Clean up resources and shut down WDBX components.

        This method should be called before the program exits to ensure
        all resources are properly released.
        """
        logger.debug("Shutting down WDBX")

        # Shut down plugins
        if self.enable_plugins:
            shutdown_tasks = [plugin.shutdown() for plugin in self.plugins.values()]
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks)

        # Shut down vector store
        await self.vector_store.shutdown()

        # Shut down distributed components
        if self.enable_distributed and self.shard_manager:
            await self.shard_manager.shutdown()

        logger.info("WDBX shutdown complete")

    def get_plugin(self, plugin_name: str) -> Optional[WDBXPlugin]:
        """
        Get a plugin by name.

        Args:
            plugin_name: Name of the plugin to retrieve

        Returns:
            The plugin instance if found, None otherwise
        """
        if not self.enable_plugins:
            logger.warning("Plugin system is disabled, cannot get plugin")
            return None

        plugin = self.plugins.get(plugin_name)
        if plugin is None:
            logger.warning(f"Plugin not found: {plugin_name}")

        return plugin

    def register_plugin(self, plugin: WDBXPlugin) -> bool:
        """
        Register a new plugin with WDBX.

        Args:
            plugin: Plugin instance to register

        Returns:
            True if the plugin was registered successfully, False otherwise
        """
        if not self.enable_plugins:
            logger.warning("Plugin system is disabled, cannot register plugin")
            return False

        if plugin.name in self.plugins:
            logger.warning(f"Plugin already registered: {plugin.name}")
            return False

        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
        return True

    def vector_store(
        self,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Store a vector in the database.

        Args:
            vector: Vector data to store
            metadata: Optional metadata associated with the vector
            id: Optional custom ID for the vector (will be generated if not provided)

        Returns:
            ID of the stored vector
        """
        # Validate vector dimensions
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dim}, got {len(vector)}"
            )

        # Generate ID if not provided
        vector_id = id or str(uuid.uuid4())

        # Store the vector
        self.vector_store.store(vector_id, vector, metadata)

        return vector_id

    async def vector_store_async(
        self,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """
        Asynchronously store a vector in the database.

        Args:
            vector: Vector data to store
            metadata: Optional metadata associated with the vector
            id: Optional custom ID for the vector (will be generated if not provided)

        Returns:
            ID of the stored vector
        """
        # Validate vector dimensions
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dim}, got {len(vector)}"
            )

        # Generate ID if not provided
        vector_id = id or str(uuid.uuid4())

        # Store the vector
        await self.vector_store.store_async(vector_id, vector, metadata)

        return vector_id

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for vectors similar to the query vector.

        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            filter_metadata: Optional metadata filters for the search

        Returns:
            List of tuples containing (vector_id, similarity_score, metadata)
        """
        # Validate vector dimensions
        if len(query_vector) != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dim}, got {len(query_vector)}"
            )

        # Perform the search
        results = self.vector_store.search(
            query_vector,
            limit=limit,
            threshold=threshold,
            filter_metadata=filter_metadata,
        )

        return results

    async def vector_search_async(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Asynchronously search for vectors similar to the query vector.

        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            filter_metadata: Optional metadata filters for the search

        Returns:
            List of tuples containing (vector_id, similarity_score, metadata)
        """
        # Validate vector dimensions
        if len(query_vector) != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.vector_dim}, got {len(query_vector)}"
            )

        # Perform the search
        results = await self.vector_store.search_async(
            query_vector,
            limit=limit,
            threshold=threshold,
            filter_metadata=filter_metadata,
        )

        return results

    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the database.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if the vector was deleted, False if it wasn't found
        """
        return self.vector_store.delete(vector_id)

    async def delete_vector_async(self, vector_id: str) -> bool:
        """
        Asynchronously delete a vector from the database.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if the vector was deleted, False if it wasn't found
        """
        return await self.vector_store.delete_async(vector_id)

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a vector.

        Args:
            vector_id: ID of the vector to update
            metadata: New metadata to associate with the vector

        Returns:
            True if the metadata was updated, False if the vector wasn't found
        """
        return self.vector_store.update_metadata(vector_id, metadata)

    async def update_metadata_async(
        self, vector_id: str, metadata: Dict[str, Any]
    ) -> bool:
        """
        Asynchronously update the metadata for a vector.

        Args:
            vector_id: ID of the vector to update
            metadata: New metadata to associate with the vector

        Returns:
            True if the metadata was updated, False if the vector wasn't found
        """
        return await self.vector_store.update_metadata_async(vector_id, metadata)

    def get_vector(
        self, vector_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Get a vector and its metadata by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        return self.vector_store.get(vector_id)

    async def get_vector_async(
        self, vector_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Asynchronously get a vector and its metadata by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        return await self.vector_store.get_async(vector_id)

    def count_vectors(self) -> int:
        """
        Count the total number of vectors in the database.

        Returns:
            Total number of vectors
        """
        return self.vector_store.count()

    def clear(self) -> int:
        """
        Clear all vectors from the database.

        Returns:
            Number of vectors removed
        """
        return self.vector_store.clear()

    async def clear_async(self) -> int:
        """
        Asynchronously clear all vectors from the database.

        Returns:
            Number of vectors removed
        """
        return await self.vector_store.clear_async()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "version": self.version,
            "vector_dimension": self.vector_dim,
            "num_shards": self.num_shards,
            "total_vectors": self.count_vectors(),
            "plugins_enabled": self.enable_plugins,
            "plugins_loaded": len(self.plugins) if self.enable_plugins else 0,
            "distributed_enabled": self.enable_distributed,
            "gpu_enabled": self.enable_gpu,
        }

        # Add vector store stats
        vector_store_stats = self.vector_store.get_stats()
        stats.update(vector_store_stats)

        return stats
