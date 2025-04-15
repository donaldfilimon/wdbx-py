"""
Vector storage implementation for WDBX.
"""

import os
import json
import logging
import asyncio
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

from .config import WDBXConfig
from .indexing import HNSWIndex, FaissIndex

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector storage for WDBX.

    This class handles the storage and retrieval of vectors and associated metadata.
    It supports various indexing methods for efficient similarity search.

    Attributes:
        vector_dim (int): Dimension of vectors stored
        data_dir (Path): Directory for storing vector data
        num_shards (int): Number of shards for distributed storage
        use_gpu (bool): Whether to use GPU acceleration if available
        index_type (str): Type of index to use (hnsw, faiss)
        config (WDBXConfig): Configuration object
    """

    def __init__(
        self,
        vector_dim: int,
        data_dir: Path,
        num_shards: int = 1,
        use_gpu: bool = False,
        index_type: str = "hnsw",
        config: Optional[WDBXConfig] = None,
    ):
        """
        Initialize the vector store.

        Args:
            vector_dim: Dimension of vectors stored
            data_dir: Directory for storing vector data
            num_shards: Number of shards for distributed storage
            use_gpu: Whether to use GPU acceleration if available
            index_type: Type of index to use (hnsw, faiss)
            config: Configuration object
        """
        self.vector_dim = vector_dim
        self.data_dir = data_dir
        self.num_shards = num_shards
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.config = config or WDBXConfig({})

        # Initialize data structures
        self.vectors = {}  # id -> vector
        self.metadata = {}  # id -> metadata
        self.indices = []  # List of indices (one per shard)

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get(
                "VECTOR_STORE_THREADS", os.cpu_count() or 4))

        # Create data directories
        self._create_dirs()

        # Initialize indices
        self._init_indices()

        # Load existing data if any
        self._load_data()

        logger.info(
            f"VectorStore initialized with {self.count()} vectors, {self.num_shards} shards, index_type={self.index_type}")

    def _create_dirs(self):
        """Create necessary data directories."""
        # Main data directory
        vectors_dir = self.data_dir / "vectors"
        if not vectors_dir.exists():
            vectors_dir.mkdir(parents=True)

        # Metadata directory
        metadata_dir = self.data_dir / "metadata"
        if not metadata_dir.exists():
            metadata_dir.mkdir(parents=True)

        # Indices directory
        indices_dir = self.data_dir / "indices"
        if not indices_dir.exists():
            indices_dir.mkdir(parents=True)

        # Shard directories
        for shard in range(self.num_shards):
            shard_dir = self.data_dir / f"shard_{shard}"
            if not shard_dir.exists():
                shard_dir.mkdir(parents=True)

    def _init_indices(self):
        """Initialize vector indices for each shard."""
        self.indices = []

        for shard in range(self.num_shards):
            index_path = self.data_dir / f"shard_{shard}" / "index"

            if self.index_type == "hnsw":
                index = HNSWIndex(
                    vector_dim=self.vector_dim,
                    index_path=index_path,
                    config=self.config
                )
            elif self.index_type == "faiss":
                index = FaissIndex(
                    vector_dim=self.vector_dim,
                    index_path=index_path,
                    use_gpu=self.use_gpu,
                    config=self.config
                )
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            self.indices.append(index)

    def _load_data(self):
        """Load existing vector data from disk."""
        # Load metadata
        metadata_path = self.data_dir / "metadata" / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug(
                    f"Loaded metadata for {len(self.metadata)} vectors")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

        # Load vectors
        vectors_path = self.data_dir / "vectors" / "vectors.pickle"
        if vectors_path.exists():
            try:
                with open(vectors_path, 'rb') as f:
                    self.vectors = pickle.load(f)
                logger.debug(f"Loaded {len(self.vectors)} vectors")
            except Exception as e:
                logger.error(f"Error loading vectors: {e}")

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = self.data_dir / "metadata" / "metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f)
            logger.debug(f"Saved metadata for {len(self.metadata)} vectors")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def _save_vectors(self):
        """Save vectors to disk."""
        vectors_path = self.data_dir / "vectors" / "vectors.pickle"
        try:
            with open(vectors_path, 'wb') as f:
                pickle.dump(self.vectors, f)
            logger.debug(f"Saved {len(self.vectors)} vectors")
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")

    def _get_shard_for_id(self, vector_id: str) -> int:
        """
        Determine which shard a vector ID should go to.

        Args:
            vector_id: ID of the vector

        Returns:
            Shard index for the vector
        """
        # Simple hash-based sharding
        hash_val = hash(vector_id)
        return abs(hash_val) % self.num_shards

    async def initialize(self):
        """Asynchronously initialize the vector store."""
        # Initialize each index
        init_tasks = []
        for index in self.indices:
            init_tasks.append(index.initialize())

        if init_tasks:
            await asyncio.gather(*init_tasks)

    async def shutdown(self):
        """Clean up resources and save data."""
        # Save data
        self._save_metadata()
        self._save_vectors()

        # Shutdown indices
        shutdown_tasks = []
        for index in self.indices:
            shutdown_tasks.append(index.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks)

        # Shutdown thread pool
        self.thread_pool.shutdown()

    def store(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a vector and its metadata.

        Args:
            vector_id: ID for the vector
            vector: Vector data
            metadata: Optional metadata associated with the vector

        Returns:
            True if the storage was successful
        """
        try:
            # Convert vector to numpy array
            vector_np = np.array(vector, dtype=np.float32)

            # Store vector and metadata
            self.vectors[vector_id] = vector_np
            self.metadata[vector_id] = metadata or {}

            # Add to appropriate index
            shard = self._get_shard_for_id(vector_id)
            self.indices[shard].add(vector_id, vector_np)

            # If configured to save immediately, do so
            if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
                self._save_metadata()
                self._save_vectors()

            return True
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            return False

    async def store_async(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Asynchronously store a vector and its metadata.

        Args:
            vector_id: ID for the vector
            vector: Vector data
            metadata: Optional metadata associated with the vector

        Returns:
            True if the storage was successful
        """
        try:
            # Convert vector to numpy array
            vector_np = np.array(vector, dtype=np.float32)

            # Store vector and metadata
            self.vectors[vector_id] = vector_np
            self.metadata[vector_id] = metadata or {}

            # Add to appropriate index asynchronously
            shard = self._get_shard_for_id(vector_id)
            await self.indices[shard].add_async(vector_id, vector_np)

            # If configured to save immediately, do so
            if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.thread_pool,
                    lambda: (self._save_metadata(), self._save_vectors())
                )

            return True
        except Exception as e:
            logger.error(f"Error storing vector asynchronously: {e}")
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            filter_metadata: Optional metadata filters for the search

        Returns:
            List of tuples containing (vector_id, similarity_score, metadata)
        """
        # Convert query vector to numpy array
        query_np = np.array(query_vector, dtype=np.float32)

        # Search in each shard and collect results
        all_results = []
        for index in self.indices:
            results = index.search(query_np, limit=limit)
            all_results.extend(results)

        # Sort by similarity score and apply limit
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold
        if threshold > 0:
            all_results = [r for r in all_results if r[1] >= threshold]

        # Apply metadata filter if specified
        if filter_metadata:
            filtered_results = []
            for vector_id, score in all_results:
                if self._matches_filter(vector_id, filter_metadata):
                    filtered_results.append((vector_id, score))
            all_results = filtered_results

        # Limit results
        all_results = all_results[:limit]

        # Add metadata to results
        results_with_metadata = [
            (vector_id, score, self.metadata.get(vector_id, {}))
            for vector_id, score in all_results
        ]

        return results_with_metadata

    async def search_async(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Asynchronously search for similar vectors.

        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            filter_metadata: Optional metadata filters for the search

        Returns:
            List of tuples containing (vector_id, similarity_score, metadata)
        """
        # Convert query vector to numpy array
        query_np = np.array(query_vector, dtype=np.float32)

        # Search in each shard asynchronously and collect results
        search_tasks = [index.search_async(
            query_np, limit=limit) for index in self.indices]
        shard_results = await asyncio.gather(*search_tasks)

        # Combine results from all shards
        all_results = []
        for results in shard_results:
            all_results.extend(results)

        # Sort by similarity score and apply limit
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold
        if threshold > 0:
            all_results = [r for r in all_results if r[1] >= threshold]

        # Apply metadata filter if specified
        if filter_metadata:
            filtered_results = []
            for vector_id, score in all_results:
                if self._matches_filter(vector_id, filter_metadata):
                    filtered_results.append((vector_id, score))
            all_results = filtered_results

        # Limit results
        all_results = all_results[:limit]

        # Add metadata to results
        results_with_metadata = [
            (vector_id, score, self.metadata.get(vector_id, {}))
            for vector_id, score in all_results
        ]

        return results_with_metadata

    def _matches_filter(
            self, vector_id: str, filter_metadata: Dict[str, Any]) -> bool:
        """
        Check if a vector's metadata matches the filter criteria.

        Args:
            vector_id: ID of the vector to check
            filter_metadata: Metadata filter criteria

        Returns:
            True if the vector's metadata matches the filter, False otherwise
        """
        # Get the vector's metadata
        metadata = self.metadata.get(vector_id, {})

        # Check each filter criterion
        for key, value in filter_metadata.items():
            # Handle special operators (e.g., $gt, $lt, $in)
            if isinstance(
                    value, dict) and list(
                    value.keys())[0].startswith('$'):
                op = list(value.keys())[0]
                op_value = value[op]

                if op == '$gt':
                    if key not in metadata or metadata[key] <= op_value:
                        return False
                elif op == '$lt':
                    if key not in metadata or metadata[key] >= op_value:
                        return False
                elif op == '$gte':
                    if key not in metadata or metadata[key] < op_value:
                        return False
                elif op == '$lte':
                    if key not in metadata or metadata[key] > op_value:
                        return False
                elif op == '$in':
                    if key not in metadata or metadata[key] not in op_value:
                        return False
                elif op == '$nin':
                    if key in metadata and metadata[key] in op_value:
                        return False
                elif op == '$exists':
                    if op_value and key not in metadata:
                        return False
                    if not op_value and key in metadata:
                        return False
            # Regular equality check
            else:
                if key not in metadata or metadata[key] != value:
                    return False

        return True

    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector and its metadata.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if the vector was deleted, False if it wasn't found
        """
        # Check if vector exists
        if vector_id not in self.vectors:
            return False

        # Remove from appropriate index
        shard = self._get_shard_for_id(vector_id)
        self.indices[shard].remove(vector_id)

        # Remove from memory
        self.vectors.pop(vector_id, None)
        self.metadata.pop(vector_id, None)

        # If configured to save immediately, do so
        if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
            self._save_metadata()
            self._save_vectors()

        return True

    async def delete_async(self, vector_id: str) -> bool:
        """
        Asynchronously delete a vector and its metadata.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if the vector was deleted, False if it wasn't found
        """
        # Check if vector exists
        if vector_id not in self.vectors:
            return False

        # Remove from appropriate index asynchronously
        shard = self._get_shard_for_id(vector_id)
        await self.indices[shard].remove_async(vector_id)

        # Remove from memory
        self.vectors.pop(vector_id, None)
        self.metadata.pop(vector_id, None)

        # If configured to save immediately, do so
        if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                lambda: (self._save_metadata(), self._save_vectors())
            )

        return True

    def update_metadata(
            self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a vector.

        Args:
            vector_id: ID of the vector to update
            metadata: New metadata to associate with the vector

        Returns:
            True if the metadata was updated, False if the vector wasn't found
        """
        # Check if vector exists
        if vector_id not in self.vectors:
            return False

        # Update metadata
        self.metadata[vector_id] = metadata

        # If configured to save immediately, do so
        if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
            self._save_metadata()

        return True

    async def update_metadata_async(
            self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Asynchronously update the metadata for a vector.

        Args:
            vector_id: ID of the vector to update
            metadata: New metadata to associate with the vector

        Returns:
            True if the metadata was updated, False if the vector wasn't found
        """
        # Check if vector exists
        if vector_id not in self.vectors:
            return False

        # Update metadata
        self.metadata[vector_id] = metadata

        # If configured to save immediately, do so
        if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                self._save_metadata
            )

        return True

    def get(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Get a vector and its metadata by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        # Check if vector exists
        if vector_id not in self.vectors:
            return None

        # Get vector and metadata
        vector = self.vectors[vector_id].tolist()
        metadata = self.metadata.get(vector_id, {})

        return vector, metadata

    async def get_async(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Asynchronously get a vector and its metadata by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None otherwise
        """
        # We can use the synchronous version since it's fast enough
        return self.get(vector_id)

    def count(self) -> int:
        """
        Count the total number of vectors in the database.

        Returns:
            Total number of vectors
        """
        return len(self.vectors)

    def clear(self) -> int:
        """
        Clear all vectors from the database.

        Returns:
            Number of vectors removed
        """
        count = len(self.vectors)

        # Clear indices
        for index in self.indices:
            index.clear()

        # Clear memory
        self.vectors = {}
        self.metadata = {}

        # Save empty state
        self._save_metadata()
        self._save_vectors()

        return count

    async def clear_async(self) -> int:
        """
        Asynchronously clear all vectors from the database.

        Returns:
            Number of vectors removed
        """
        count = len(self.vectors)

        # Clear indices asynchronously
        clear_tasks = [index.clear_async() for index in self.indices]
        await asyncio.gather(*clear_tasks)

        # Clear memory
        self.vectors = {}
        self.metadata = {}

        # Save empty state in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_pool,
            lambda: (self._save_metadata(), self._save_vectors())
        )

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary of statistics
        """
        index_stats = []
        for i, index in enumerate(self.indices):
            index_stats.append({
                "shard": i,
                "type": self.index_type,
                "size": index.size(),
                "stats": index.get_stats()
            })

        return {
            "vector_count": len(self.vectors),
            "metadata_count": len(self.metadata),
            "index_type": self.index_type,
            "num_shards": self.num_shards,
            "vector_dim": self.vector_dim,
            "use_gpu": self.use_gpu,
            "indices": index_stats,
        }

    def optimize(self) -> bool:
        """
        Optimize the vector indices for better performance.

        Returns:
            True if optimization was successful
        """
        for index in self.indices:
            index.optimize()
        return True

    async def optimize_async() -> bool:
        """
        Asynchronously optimize the vector indices for better performance.

        Returns:
            True if optimization was successful
        """
        optimize_tasks = [index.optimize_async() for index in self.indices]
        await asyncio.gather(*optimize_tasks)
        return True

    def batch_store(
        self,
        vectors: Dict[str, List[float]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> int:
        """
        Store multiple vectors in batch.

        Args:
            vectors: Dictionary mapping vector IDs to vectors
            metadata: Optional dictionary mapping vector IDs to metadata

        Returns:
            Number of vectors stored
        """
        # Default empty metadata if not provided
        metadata = metadata or {}

        # Group vectors by shard
        shard_vectors = {}
        for vector_id, vector in vectors.items():
            # Convert to numpy array
            vector_np = np.array(vector, dtype=np.float32)

            # Store in memory
            self.vectors[vector_id] = vector_np
            self.metadata[vector_id] = metadata.get(vector_id, {})

            # Group by shard
            shard = self._get_shard_for_id(vector_id)
            if shard not in shard_vectors:
                shard_vectors[shard] = {}
            shard_vectors[shard][vector_id] = vector_np

        # Add to indices
        for shard, shard_vecs in shard_vectors.items():
            self.indices[shard].batch_add(shard_vecs)

        # Save if configured to do so
        if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
            self._save_metadata()
            self._save_vectors()

        return len(vectors)

    async def batch_store_async(
        self,
        vectors: Dict[str, List[float]],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> int:
        """
        Asynchronously store multiple vectors in batch.

        Args:
            vectors: Dictionary mapping vector IDs to vectors
            metadata: Optional dictionary mapping vector IDs to metadata

        Returns:
            Number of vectors stored
        """
        # Default empty metadata if not provided
        metadata = metadata or {}

        # Group vectors by shard
        shard_vectors = {}
        for vector_id, vector in vectors.items():
            # Convert to numpy array
            vector_np = np.array(vector, dtype=np.float32)

            # Store in memory
            self.vectors[vector_id] = vector_np
            self.metadata[vector_id] = metadata.get(vector_id, {})

            # Group by shard
            shard = self._get_shard_for_id(vector_id)
            if shard not in shard_vectors:
                shard_vectors[shard] = {}
            shard_vectors[shard][vector_id] = vector_np

        # Add to indices asynchronously
        batch_tasks = []
        for shard, shard_vecs in shard_vectors.items():
            batch_tasks.append(self.indices[shard].batch_add_async(shard_vecs))

        if batch_tasks:
            await asyncio.gather(*batch_tasks)

        # Save if configured to do so
        if self.config.get("VECTOR_STORE_SAVE_IMMEDIATELY", False):
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                lambda: (self._save_metadata(), self._save_vectors())
            )

        return len(vectors)
