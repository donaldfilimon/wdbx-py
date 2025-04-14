"""
Unit tests for WDBX core functionality.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import numpy as np

from wdbx import WDBX
from wdbx.core.config import WDBXConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        "WDBX_TEST_OPTION": "test_value",
        "WDBX_VECTOR_STORE_SAVE_IMMEDIATELY": False,
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def wdbx_instance(temp_dir, config):
    """Create and initialize a WDBX instance for testing."""
    wdbx = WDBX(
        vector_dimension=4,  # Small dimension for testing
        num_shards=2,
        data_dir=temp_dir,
        config=config,
        enable_plugins=False,  # Disable plugins for core tests
    )

    # Initialize
    await wdbx.initialize()

    yield wdbx

    # Clean up
    await wdbx.shutdown()

# Test configuration


def test_config_creation():
    """Test creating a configuration object."""
    config = WDBXConfig({"WDBX_TEST_OPTION": "test_value"})
    assert config.get("WDBX_TEST_OPTION") == "test_value"
    assert config.get("NONEXISTENT", "default") == "default"


def test_config_environment(monkeypatch):
    """Test loading configuration from environment variables."""
    monkeypatch.setenv("WDBX_ENV_TEST", "env_value")
    config = WDBXConfig()
    assert config.get("WDBX_ENV_TEST") == "env_value"


def test_config_type_conversion():
    """Test configuration type conversion."""
    config = WDBXConfig({
        "WDBX_INT_OPTION": "42",
        "WDBX_FLOAT_OPTION": "3.14",
        "WDBX_BOOL_OPTION": "true",
        "WDBX_LIST_OPTION": "[1, 2, 3]",
    })

    assert config.get_typed("WDBX_INT_OPTION", int) == 42
    assert config.get_typed("WDBX_FLOAT_OPTION", float) == 3.14
    assert config.get_typed("WDBX_BOOL_OPTION", bool) is True
    assert config.get_typed("WDBX_LIST_OPTION", list) == [1, 2, 3]

    # Test default value
    assert config.get_typed("NONEXISTENT", int, 99) == 99

# Test WDBX instance creation


def test_wdbx_creation(temp_dir, config):
    """Test creating a WDBX instance."""
    wdbx = WDBX(
        vector_dimension=4,
        num_shards=2,
        data_dir=temp_dir,
        config=config,
    )

    assert wdbx.vector_dim == 4
    assert wdbx.num_shards == 2
    assert wdbx.data_dir == Path(temp_dir)
    assert wdbx.config.get("WDBX_TEST_OPTION") == "test_value"

# Test vector operations


@pytest.mark.asyncio
async def test_vector_store_and_search(wdbx_instance):
    """Test storing and searching vectors."""
    # Store a vector
    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {"source": "test", "content": "test vector"}
    vector_id = wdbx_instance.vector_store(vector, metadata)

    # Verify vector was stored
    assert wdbx_instance.count_vectors() == 1

    # Get the vector
    result = wdbx_instance.get_vector(vector_id)
    assert result is not None
    stored_vector, stored_metadata = result

    # Verify vector and metadata
    assert len(stored_vector) == 4
    assert np.allclose(stored_vector, vector)
    assert stored_metadata["source"] == "test"
    assert stored_metadata["content"] == "test vector"

    # Search for similar vectors
    search_results = wdbx_instance.vector_search(vector, limit=5)
    assert len(search_results) == 1
    found_id, similarity, found_metadata = search_results[0]

    # Verify search results
    assert found_id == vector_id
    assert similarity > 0.99  # Should be very similar to itself
    assert found_metadata["source"] == "test"


@pytest.mark.asyncio
async def test_vector_async_operations(wdbx_instance):
    """Test asynchronous vector operations."""
    # Store a vector asynchronously
    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {"source": "async_test", "content": "async test vector"}
    vector_id = await wdbx_instance.vector_store_async(vector, metadata)

    # Verify vector was stored
    assert wdbx_instance.count_vectors() == 1

    # Get the vector asynchronously
    result = await wdbx_instance.get_vector_async(vector_id)
    assert result is not None
    stored_vector, stored_metadata = result

    # Verify vector and metadata
    assert len(stored_vector) == 4
    assert np.allclose(stored_vector, vector)
    assert stored_metadata["source"] == "async_test"

    # Search for similar vectors asynchronously
    search_results = await wdbx_instance.vector_search_async(vector, limit=5)
    assert len(search_results) == 1
    found_id, similarity, found_metadata = search_results[0]

    # Verify search results
    assert found_id == vector_id
    assert similarity > 0.99
    assert found_metadata["source"] == "async_test"

    # Update metadata asynchronously
    new_metadata = {"source": "updated", "content": "updated test vector"}
    success = await wdbx_instance.update_metadata_async(vector_id, new_metadata)
    assert success is True

    # Verify metadata was updated
    result = await wdbx_instance.get_vector_async(vector_id)
    assert result is not None
    _, updated_metadata = result
    assert updated_metadata["source"] == "updated"

    # Delete vector asynchronously
    success = await wdbx_instance.delete_vector_async(vector_id)
    assert success is True

    # Verify vector was deleted
    assert wdbx_instance.count_vectors() == 0


@pytest.mark.asyncio
async def test_vector_batch_operations(wdbx_instance):
    """Test batch vector operations."""
    # Create multiple vectors
    vectors = {}
    for i in range(10):
        vector = [i/10, (i+1)/10, (i+2)/10, (i+3)/10]
        vectors[f"vec_{i}"] = vector

    # Create metadata for each vector
    metadata = {
        vector_id: {"index": i, "source": "batch_test"}
        for i, vector_id in enumerate(vectors.keys())
    }

    # Store vectors in batch
    count = wdbx_instance.vector_store.batch_store(vectors, metadata)
    assert count == 10

    # Verify all vectors were stored
    assert wdbx_instance.count_vectors() == 10

    # Search for a specific vector
    search_vector = [0.5, 0.6, 0.7, 0.8]  # Should be closest to vec_5
    results = wdbx_instance.vector_search(search_vector, limit=1)
    assert len(results) == 1
    vector_id, _, _ = results[0]
    assert vector_id == "vec_5"

    # Test metadata filtering
    filtered_results = wdbx_instance.vector_search(
        search_vector,
        limit=10,
        filter_metadata={"index": {"$lt": 3}}
    )
    assert len(filtered_results) == 3
    for _, _, meta in filtered_results:
        assert meta["index"] < 3

    # Clear all vectors
    count = wdbx_instance.clear()
    assert count == 10
    assert wdbx_instance.count_vectors() == 0

# Test error handling


@pytest.mark.asyncio
async def test_error_handling(wdbx_instance):
    """Test error handling in vector operations."""
    # Test vector dimension mismatch
    wrong_dim_vector = [0.1, 0.2, 0.3]  # 3D instead of 4D
    with pytest.raises(ValueError, match="dimension mismatch"):
        wdbx_instance.vector_store(wrong_dim_vector, {})

    # Test getting nonexistent vector
    nonexistent = wdbx_instance.get_vector("nonexistent_id")
    assert nonexistent is None

    # Test deleting nonexistent vector
    success = wdbx_instance.delete_vector("nonexistent_id")
    assert success is False

    # Test updating nonexistent metadata
    success = wdbx_instance.update_metadata(
        "nonexistent_id", {"test": "value"})
    assert success is False

# Test persistence


@pytest.mark.asyncio
async def test_persistence(temp_dir, config):
    """Test persistence of stored vectors."""
    # Create and initialize a WDBX instance
    wdbx1 = WDBX(
        vector_dimension=4,
        num_shards=2,
        data_dir=temp_dir,
        config=config,
    )
    await wdbx1.initialize()

    # Store a vector
    vector = [0.1, 0.2, 0.3, 0.4]
    metadata = {"source": "persistence_test"}
    vector_id = wdbx1.vector_store(vector, metadata)

    # Force save
    wdbx1.vector_store._save_metadata()
    wdbx1.vector_store._save_vectors()

    # Shutdown
    await wdbx1.shutdown()

    # Create a new instance with the same data directory
    wdbx2 = WDBX(
        vector_dimension=4,
        num_shards=2,
        data_dir=temp_dir,
        config=config,
    )
    await wdbx2.initialize()

    # Verify vector was loaded
    assert wdbx2.count_vectors() == 1

    # Get the vector
    result = wdbx2.get_vector(vector_id)
    assert result is not None
    stored_vector, stored_metadata = result

    # Verify vector and metadata
    assert len(stored_vector) == 4
    assert np.allclose(stored_vector, vector)
    assert stored_metadata["source"] == "persistence_test"

    # Clean up
    await wdbx2.shutdown()

# Test statistics


@pytest.mark.asyncio
async def test_statistics(wdbx_instance):
    """Test getting statistics about the database."""
    # Store some vectors
    for i in range(5):
        vector = [i/10, (i+1)/10, (i+2)/10, (i+3)/10]
        metadata = {"index": i}
        wdbx_instance.vector_store(vector, metadata)

    # Get statistics
    stats = wdbx_instance.get_stats()

    # Verify statistics
    assert stats["vector_dimension"] == 4
    assert stats["num_shards"] == 2
    assert stats["total_vectors"] == 5
    assert "version" in stats

    # Check vector store stats
    assert "vector_count" in stats
    assert stats["vector_count"] == 5
    assert "index_type" in stats
    assert "indices" in stats
    assert len(stats["indices"]) == 2  # 2 shards
