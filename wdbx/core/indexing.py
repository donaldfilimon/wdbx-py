"""
Vector indexing implementations for WDBX.
"""

import os
import pickle
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class VectorIndex(ABC):
    """
    Abstract base class for vector indices.

    This class defines the interface that all vector index implementations must follow.
    """

    @abstractmethod
    def __init__(self, vector_dim: int, index_path: Path, config: Any = None):
        """
        Initialize the vector index.

        Args:
            vector_dim: Dimension of vectors in the index
            index_path: Path to store the index
            config: Optional configuration
        """
        pass

    @abstractmethod
    async def initialize(self):
        """Asynchronously initialize the index."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Asynchronously clean up resources."""
        pass

    @abstractmethod
    def add(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Add a vector to the index.

        Args:
            vector_id: ID for the vector
            vector: The vector to add

        Returns:
            True if the vector was added successfully
        """
        pass

    @abstractmethod
    async def add_async(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Asynchronously add a vector to the index.

        Args:
            vector_id: ID for the vector
            vector: The vector to add

        Returns:
            True if the vector was added successfully
        """
        pass

    @abstractmethod
    def batch_add(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Add multiple vectors to the index.

        Args:
            vectors: Dictionary mapping vector IDs to vectors

        Returns:
            True if the vectors were added successfully
        """
        pass

    @abstractmethod
    async def batch_add_async(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Asynchronously add multiple vectors to the index.

        Args:
            vectors: Dictionary mapping vector IDs to vectors

        Returns:
            True if the vectors were added successfully
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        pass

    @abstractmethod
    async def search_async(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Asynchronously search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        pass

    @abstractmethod
    def remove(self, vector_id: str) -> bool:
        """
        Remove a vector from the index.

        Args:
            vector_id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found
        """
        pass

    @abstractmethod
    async def remove_async(self, vector_id: str) -> bool:
        """
        Asynchronously remove a vector from the index.

        Args:
            vector_id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all vectors from the index.

        Returns:
            True if the index was cleared successfully
        """
        pass

    @abstractmethod
    async def clear_async(self) -> bool:
        """
        Asynchronously clear all vectors from the index.

        Returns:
            True if the index was cleared successfully
        """
        pass

    @abstractmethod
    def optimize(self) -> bool:
        """
        Optimize the index for better performance.

        Returns:
            True if the index was optimized successfully
        """
        pass

    @abstractmethod
    async def optimize_async(self) -> bool:
        """
        Asynchronously optimize the index for better performance.

        Returns:
            True if the index was optimized successfully
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            Number of vectors in the index
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary of statistics
        """
        pass


class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World (HNSW) index implementation.

    HNSW is an efficient algorithm for approximate nearest neighbor search.
    This implementation uses the hnswlib library.
    """

    def __init__(self, vector_dim: int, index_path: Path, config: Any = None):
        """
        Initialize the HNSW index.

        Args:
            vector_dim: Dimension of vectors in the index
            index_path: Path to store the index
            config: Optional configuration
        """
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.config = config or {}

        # Configuration parameters
        self.m = self.config.get("HNSW_M", 16)
        self.ef_construction = self.config.get("HNSW_EF_CONSTRUCTION", 200)
        self.ef_search = self.config.get("HNSW_EF_SEARCH", 50)
        self.max_elements = self.config.get("HNSW_MAX_ELEMENTS", 100000)

        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Initialize index
        self._init_index()

        # Map of vector_id to internal index
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0

        # Load existing index if available
        self._load_index()

        logger.debug(
            f"Initialized HNSW index with dim={vector_dim}, M={self.m}, ef_construction={self.ef_construction}")

    def _init_index(self):
        """Initialize the HNSW index."""
        try:
            import hnswlib

            # Create the index
            self.index = hnswlib.Index(space='cosine', dim=self.vector_dim)

            # Initialize with parameters
            self.index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.m
            )

            # Set search parameters
            self.index.set_ef(self.ef_search)

        except ImportError:
            logger.error(
                "hnswlib not installed. Please install with: pip install hnswlib")
            raise

    def _load_index(self):
        """Load the index from disk if it exists."""
        index_file = self.index_path.with_suffix('.bin')
        mapping_file = self.index_path.with_suffix('.mapping')

        if index_file.exists() and mapping_file.exists():
            try:
                # Load the index
                self.index.load_index(
                    str(index_file),
                    max_elements=self.max_elements)

                # Load the ID mapping
                with open(mapping_file, 'rb') as f:
                    mapping_data = pickle.load(f)
                    self.id_to_index = mapping_data.get('id_to_index', {})
                    self.index_to_id = mapping_data.get('index_to_id', {})
                    self.next_index = mapping_data.get('next_index', 0)

                logger.debug(
                    f"Loaded HNSW index from {index_file} with {len(self.id_to_index)} vectors")
            except Exception as e:
                logger.error(f"Error loading HNSW index: {e}")
                # Reinitialize the index
                self._init_index()
                self.id_to_index = {}
                self.index_to_id = {}
                self.next_index = 0

    def _save_index(self):
        """Save the index to disk."""
        if not self.index_path.parent.exists():
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

        index_file = self.index_path.with_suffix('.bin')
        mapping_file = self.index_path.with_suffix('.mapping')

        try:
            # Save the index
            self.index.save_index(str(index_file))

            # Save the ID mapping
            with open(mapping_file, 'wb') as f:
                mapping_data = {
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'next_index': self.next_index
                }
                pickle.dump(mapping_data, f)

            logger.debug(
                f"Saved HNSW index to {index_file} with {len(self.id_to_index)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving HNSW index: {e}")
            return False

    async def initialize(self):
        """Asynchronously initialize the index."""
        # Nothing to do here, index is initialized in __init__
        pass

    async def shutdown(self):
        """Asynchronously clean up resources."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, self._save_index)
        self.thread_pool.shutdown()

    def add(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Add a vector to the index.

        Args:
            vector_id: ID for the vector
            vector: The vector to add

        Returns:
            True if the vector was added successfully
        """
        try:
            # Check if the vector already exists
            if vector_id in self.id_to_index:
                # Remove the old vector
                old_index = self.id_to_index[vector_id]
                # We can't actually remove from HNSW, but we can update it
                self.index.replace_vector(vector, old_index)
                return True

            # Add the vector to the index
            self.index.add_items(vector, self.next_index)

            # Update the mappings
            self.id_to_index[vector_id] = self.next_index
            self.index_to_id[self.next_index] = vector_id
            self.next_index += 1

            # Auto-save if the index has grown significantly
            if self.next_index % 1000 == 0:
                self._save_index()

            return True
        except Exception as e:
            logger.error(f"Error adding vector to HNSW index: {e}")
            return False

    async def add_async(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Asynchronously add a vector to the index.

        Args:
            vector_id: ID for the vector
            vector: The vector to add

        Returns:
            True if the vector was added successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.add,
            vector_id,
            vector
        )

    def batch_add(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Add multiple vectors to the index.

        Args:
            vectors: Dictionary mapping vector IDs to vectors

        Returns:
            True if the vectors were added successfully
        """
        try:
            # Collect vectors to add
            vector_ids = []
            vector_data = []
            vector_indices = []

            for vector_id, vector in vectors.items():
                # Check if the vector already exists
                if vector_id in self.id_to_index:
                    # Remove the old vector
                    old_index = self.id_to_index[vector_id]
                    # We can't actually remove from HNSW, but we can update it
                    self.index.replace_vector(vector, old_index)
                else:
                    # Add to the batch
                    vector_ids.append(vector_id)
                    vector_data.append(vector)
                    vector_indices.append(self.next_index)

                    # Update the mappings
                    self.id_to_index[vector_id] = self.next_index
                    self.index_to_id[self.next_index] = vector_id
                    self.next_index += 1

            # Add the vectors to the index if any
            if vector_data:
                self.index.add_items(np.array(vector_data), vector_indices)

            # Auto-save if the index has grown significantly
            if self.next_index % 1000 == 0:
                self._save_index()

            return True
        except Exception as e:
            logger.error(f"Error batch adding vectors to HNSW index: {e}")
            return False

    async def batch_add_async(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Asynchronously add multiple vectors to the index.

        Args:
            vectors: Dictionary mapping vector IDs to vectors

        Returns:
            True if the vectors were added successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.batch_add,
            vectors
        )

    def search(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        try:
            # Adjust limit based on index size
            actual_limit = min(limit, len(self.id_to_index))
            if actual_limit == 0:
                return []

            # Perform the search
            indices, distances = self.index.knn_query(
                query_vector, k=actual_limit)

            # Convert distances to similarities (for cosine distance)
            similarities = 1.0 - distances[0]

            # Map internal indices to vector IDs
            results = [
                (self.index_to_id.get(int(idx), str(idx)), float(sim))
                for idx, sim in zip(indices[0], similarities)
            ]

            return results
        except Exception as e:
            logger.error(f"Error searching HNSW index: {e}")
            return []

    async def search_async(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Asynchronously search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.search,
            query_vector,
            limit
        )

    def remove(self, vector_id: str) -> bool:
        """
        Remove a vector from the index.

        Note: HNSW doesn't support true removal. We just remove from our mapping.
        The vector will still be in the index but will never be returned.

        Args:
            vector_id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found
        """
        if vector_id not in self.id_to_index:
            return False

        # Get the internal index
        index = self.id_to_index[vector_id]

        # Remove from the mappings
        del self.id_to_index[vector_id]
        del self.index_to_id[index]

        # We can't actually remove from HNSW, but we can mark it as removed
        # by replacing with a zero vector that will never match anything
        try:
            zero_vector = np.zeros(self.vector_dim, dtype=np.float32)
            self.index.replace_vector(zero_vector, index)
            return True
        except Exception as e:
            logger.error(f"Error removing vector from HNSW index: {e}")
            # Restore the mappings in case of error
            self.id_to_index[vector_id] = index
            self.index_to_id[index] = vector_id
            return False

    async def remove_async(self, vector_id: str) -> bool:
        """
        Asynchronously remove a vector from the index.

        Args:
            vector_id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.remove,
            vector_id
        )

    def clear(self) -> bool:
        """
        Clear all vectors from the index.

        Returns:
            True if the index was cleared successfully
        """
        try:
            # Re-initialize the index
            self._init_index()

            # Clear the mappings
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0

            # Save empty index
            self._save_index()

            return True
        except Exception as e:
            logger.error(f"Error clearing HNSW index: {e}")
            return False

    async def clear_async(self) -> bool:
        """
        Asynchronously clear all vectors from the index.

        Returns:
            True if the index was cleared successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.clear
        )

    def optimize(self) -> bool:
        """
        Optimize the index for better performance.

        Returns:
            True if the index was optimized successfully
        """
        # HNSW doesn't support this operation
        return True

    async def optimize_async(self) -> bool:
        """
        Asynchronously optimize the index for better performance.

        Returns:
            True if the index was optimized successfully
        """
        # HNSW doesn't support this operation
        return True

    def size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            Number of vectors in the index
        """
        return len(self.id_to_index)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary of statistics
        """
        return {
            "type": "hnsw",
            "size": self.size(),
            "dimension": self.vector_dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_elements": self.max_elements
        }


class FaissIndex(VectorIndex):
    """
    FAISS vector index implementation.

    FAISS is a library for efficient similarity search and clustering of dense vectors.
    It contains algorithms that search in sets of vectors of any size, even ones that
    don't fit in RAM.
    """

    def __init__(
            self, vector_dim: int, index_path: Path, use_gpu: bool = False,
            config: Any = None):
        """
        Initialize the FAISS index.

        Args:
            vector_dim: Dimension of vectors in the index
            index_path: Path to store the index
            use_gpu: Whether to use GPU acceleration if available
            config: Optional configuration
        """
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.config = config or {}

        # Configuration parameters
        self.index_type = self.config.get("FAISS_INDEX_TYPE", "Flat")
        self.nprobe = self.config.get("FAISS_NPROBE", 8)

        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Create faiss index
        self._create_index()

        # Map of vector_id to internal index
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0

        # Load existing index if available
        self._load_index()

        logger.debug(
            f"Initialized FAISS index with dim={vector_dim}, type={self.index_type}, nprobe={self.nprobe}")

    def _create_index(self):
        """Create a FAISS index based on configuration."""
        try:
            import faiss

            # Determine the index type
            if self.index_type == "Flat":
                # Inner product (for normalized vectors this is cosine)
                self.index = faiss.IndexFlatIP(self.vector_dim)
            elif self.index_type == "IVF":
                # Create the quantizer
                quantizer = faiss.IndexFlatIP(self.vector_dim)
                # Number of centroids (adjust based on dataset size)
                nlist = max(int(self.config.get("FAISS_NLIST", 100)), 1)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.vector_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                # Need to train IVF index with some vectors
                self.index_needs_training = True
            elif self.index_type == "HNSW":
                # HNSW index in FAISS
                m = int(self.config.get("FAISS_HNSW_M", 32))
                self.index = faiss.IndexHNSWFlat(
                    self.vector_dim, m, faiss.METRIC_INNER_PRODUCT)
            else:
                # Default to flat index
                logger.warning(
                    f"Unknown FAISS index type '{self.index_type}', defaulting to Flat")
                self.index = faiss.IndexFlatIP(self.vector_dim)

            # Enable GPU if requested and available
            if self.use_gpu:
                try:
                    gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(
                        gpu_resources, 0, self.index)
                    logger.info("FAISS using GPU acceleration")
                except Exception as e:
                    logger.warning(f"Failed to use GPU for FAISS: {e}")

            # Set search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe

        except ImportError:
            logger.error(
                "FAISS not installed. Please install with: pip install faiss-cpu or faiss-gpu")
            raise

    def _load_index(self):
        """Load the index from disk if it exists."""
        index_file = self.index_path.with_suffix('.faiss')
        mapping_file = self.index_path.with_suffix('.mapping')

        if index_file.exists() and mapping_file.exists():
            try:
                import faiss

                # Load the index
                self.index = faiss.read_index(str(index_file))

                # Set search parameters
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.nprobe

                # Load the ID mapping
                with open(mapping_file, 'rb') as f:
                    mapping_data = pickle.load(f)
                    self.id_to_index = mapping_data.get('id_to_index', {})
                    self.index_to_id = mapping_data.get('index_to_id', {})
                    self.next_index = mapping_data.get('next_index', 0)

                # Enable GPU if requested and available
                if self.use_gpu:
                    try:
                        gpu_resources = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(
                            gpu_resources, 0, self.index)
                        logger.info("FAISS using GPU acceleration")
                    except Exception as e:
                        logger.warning(f"Failed to use GPU for FAISS: {e}")

                logger.debug(
                    f"Loaded FAISS index from {index_file} with {len(self.id_to_index)} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                # Reinitialize the index
                self._create_index()
                self.id_to_index = {}
                self.index_to_id = {}
                self.next_index = 0

    def _save_index(self):
        """Save the index to disk."""
        if not self.index_path.parent.exists():
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

        index_file = self.index_path.with_suffix('.faiss')
        mapping_file = self.index_path.with_suffix('.mapping')

        try:
            import faiss

            # Convert GPU index to CPU for saving
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_file))
            else:
                faiss.write_index(self.index, str(index_file))

            # Save the ID mapping
            with open(mapping_file, 'wb') as f:
                mapping_data = {
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'next_index': self.next_index
                }
                pickle.dump(mapping_data, f)

            logger.debug(
                f"Saved FAISS index to {index_file} with {len(self.id_to_index)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False

    async def initialize(self):
        """Asynchronously initialize the index."""
        # Nothing to do here, index is initialized in __init__
        pass

    async def shutdown(self):
        """Asynchronously clean up resources."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, self._save_index)
        self.thread_pool.shutdown()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def add(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Add a vector to the index.

        Args:
            vector_id: ID for the vector
            vector: The vector to add

        Returns:
            True if the vector was added successfully
        """
        try:
            # Check if index needs training
            if hasattr(
                    self, 'index_needs_training') and self.index_needs_training:
                if self.next_index > 0:
                    # We have enough vectors now, train the index
                    vectors_for_training = np.array(
                        [vector.astype(np.float32)
                         for vector in self.vectors.values()])
                    self.index.train(vectors_for_training)
                    self.index_needs_training = False
                else:
                    # We need at least one vector for training
                    # Just store it for now, will train later
                    self.vectors[vector_id] = vector
                    return True

            # Normalize vector for cosine similarity
            normalized_vector = self._normalize_vector(
                vector.astype(np.float32))

            # Add the vector to the index
            # Reshape to 2D array with a single row
            self.index.add(normalized_vector.reshape(1, -1))

            # Update the mappings
            self.id_to_index[vector_id] = self.next_index
            self.index_to_id[self.next_index] = vector_id
            self.next_index += 1

            # Auto-save if the index has grown significantly
            if self.next_index % 1000 == 0:
                self._save_index()

            return True
        except Exception as e:
            logger.error(f"Error adding vector to FAISS index: {e}")
            return False

    async def add_async(self, vector_id: str, vector: np.ndarray) -> bool:
        """
        Asynchronously add a vector to the index.

        Args:
            vector_id: ID for the vector
            vector: The vector to add

        Returns:
            True if the vector was added successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.add,
            vector_id,
            vector
        )

    def batch_add(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Add multiple vectors to the index.

        Args:
            vectors: Dictionary mapping vector IDs to vectors

        Returns:
            True if the vectors were added successfully
        """
        if not vectors:
            return True

        try:
            # Prepare vectors and IDs
            vector_ids = list(vectors.keys())
            vector_data = np.array([
                self._normalize_vector(vector.astype(np.float32))
                for vector in vectors.values()
            ])

            # Check if index needs training
            if hasattr(
                    self, 'index_needs_training') and self.index_needs_training:
                self.index.train(vector_data)
                self.index_needs_training = False

            # Add vectors to the index
            self.index.add(vector_data)

            # Update the mappings
            for i, vector_id in enumerate(vector_ids):
                self.id_to_index[vector_id] = self.next_index + i
                self.index_to_id[self.next_index + i] = vector_id

            # Update next index
            self.next_index += len(vectors)

            # Auto-save if the index has grown significantly
            if self.next_index % 1000 == 0:
                self._save_index()

            return True
        except Exception as e:
            logger.error(f"Error batch adding vectors to FAISS index: {e}")
            return False

    async def batch_add_async(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Asynchronously add multiple vectors to the index.

        Args:
            vectors: Dictionary mapping vector IDs to vectors

        Returns:
            True if the vectors were added successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.batch_add,
            vectors
        )

    def search(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        try:
            # Check if index is empty
            if self.next_index == 0:
                return []

            # Normalize query vector
            normalized_query = self._normalize_vector(
                query_vector.astype(np.float32))

            # Adjust limit based on index size
            actual_limit = min(limit, self.next_index)
            if actual_limit == 0:
                return []

            # Reshape to 2D array with a single row
            normalized_query = normalized_query.reshape(1, -1)

            # Perform the search
            distances, indices = self.index.search(
                normalized_query, actual_limit)

            # Convert distances to similarities (for inner product)
            # FAISS inner product returns higher values for better matches
            similarities = distances[0]

            # Map internal indices to vector IDs
            results = [
                (self.index_to_id.get(int(idx), str(idx)), float(sim))
                for idx, sim in zip(indices[0], similarities)
                if idx != -1  # FAISS returns -1 for padding when fewer than k results
            ]

            return results
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []

    async def search_async(self, query_vector: np.ndarray, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Asynchronously search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.search,
            query_vector,
            limit
        )

    def remove(self, vector_id: str) -> bool:
        """
        Remove a vector from the index.

        Note: FAISS doesn't support efficient removal. We mark it as removed in our mapping.

        Args:
            vector_id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found
        """
        if vector_id not in self.id_to_index:
            return False

        # Get the internal index
        index = self.id_to_index[vector_id]

        # Remove from the mappings
        del self.id_to_index[vector_id]
        del self.index_to_id[index]

        # We can't actually remove from FAISS efficiently, just update the mappings
        return True

    async def remove_async(self, vector_id: str) -> bool:
        """
        Asynchronously remove a vector from the index.

        Args:
            vector_id: ID of the vector to remove

        Returns:
            True if the vector was removed, False if it wasn't found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.remove,
            vector_id
        )

    def clear(self) -> bool:
        """
        Clear all vectors from the index.

        Returns:
            True if the index was cleared successfully
        """
        try:
            # Re-initialize the index
            self._create_index()

            # Clear the mappings
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0

            # Save empty index
            self._save_index()

            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {e}")
            return False

    async def clear_async(self) -> bool:
        """
        Asynchronously clear all vectors from the index.

        Returns:
            True if the index was cleared successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.clear
        )

    def optimize(self) -> bool:
        """
        Optimize the index for better performance.

        Returns:
            True if the index was optimized successfully
        """
        # Most FAISS indices don't need optimization
        # For IVF indices, we could retrain the quantizer
        if hasattr(
                self.index, 'train') and hasattr(
                self.index, 'is_trained') and not self.index.is_trained:
            try:
                # Get all vectors
                vectors = np.array(
                    [vector.astype(np.float32)
                     for vector in self.vectors.values()])
                if len(vectors) > 0:
                    self.index.train(vectors)
                    return True
            except Exception as e:
                logger.error(f"Error optimizing FAISS index: {e}")
                return False
        return True

    async def optimize_async(self) -> bool:
        """
        Asynchronously optimize the index for better performance.

        Returns:
            True if the index was optimized successfully
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self.optimize
        )

    def size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            Number of vectors in the index
        """
        return len(self.id_to_index)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary of statistics
        """
        return {
            "type": f"faiss_{self.index_type.lower()}",
            "size": self.size(),
            "dimension": self.vector_dim,
            "gpu_enabled": self.use_gpu,
            "nprobe": self.nprobe if hasattr(self.index, 'nprobe') else None
        }
