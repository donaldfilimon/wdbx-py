"""
Basic usage example for WDBX.
"""

import asyncio
from wdbx import WDBX


async def main():
    """Run a basic WDBX example."""
    # Create a WDBX instance
    wdbx = WDBX(
        vector_dimension=384,  # Common dimension for modern embedding models
        num_shards=1,
        data_dir="./wdbx_data",
        enable_plugins=True,
    )

    # Initialize
    await wdbx.initialize()

    try:
        # Store a vector
        vector = [0.1] * 384  # Example vector
        metadata = {"source": "example",
                    "content": "Sample text for demonstration"}
        vector_id = await wdbx.vector_store_async(vector, metadata)
        print(f"Stored vector with ID: {vector_id}")

        # Search for similar vectors
        results = await wdbx.vector_search_async(vector, limit=5)
        print(f"Found {len(results)} results:")
        for i, (id, similarity, meta) in enumerate(results):
            print(f"{i+1}. ID: {id}, Similarity: {similarity:.4f}")
            print(f"   Content: {meta.get('content', 'N/A')}")

        # Get database stats
        stats = wdbx.get_stats()
        print("\nDatabase statistics:")
        print(f"Total vectors: {stats['total_vectors']}")
        print(f"Vector dimension: {stats['vector_dimension']}")

        # List plugins
        print("\nLoaded plugins:")
        for plugin_name, plugin in wdbx.plugins.items():
            print(f"- {plugin_name} v{plugin.version}")

    finally:
        # Clean up
        await wdbx.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
