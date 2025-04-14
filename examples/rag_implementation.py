"""
RAG (Retrieval-Augmented Generation) implementation example using WDBX.
"""

import asyncio
import argparse
from typing import List, Dict, Any
from wdbx import WDBX


async def rag_pipeline(
    query: str,
    wdbx: WDBX,
    num_results: int = 5,
    model: str = "llama3"
) -> str:
    """
    Implement a basic RAG pipeline using WDBX.

    Args:
        query: User question or query
        wdbx: WDBX instance
        num_results: Number of documents to retrieve
        model: LLM model to use

    Returns:
        Generated response
    """
    # Step 1: Create embedding for the query
    embedding_plugin = None
    for plugin_name in ["ollama", "lmstudio", "openai"]:
        if plugin_name in wdbx.plugins:
            embedding_plugin = wdbx.plugins[plugin_name]
            break

    if not embedding_plugin:
        raise ValueError("No embedding plugin available")

    # Generate query embedding
    query_embedding = await embedding_plugin.create_embedding(query)

    # Step 2: Retrieve relevant documents
    search_results = await wdbx.vector_search_async(
        query_embedding,
        limit=num_results,
        threshold=0.6  # Minimum similarity threshold
    )

    # Step 3: Create context from retrieved documents
    context_parts = []
    for i, (_, similarity, metadata) in enumerate(search_results):
        if "content" in metadata:
            context_parts.append(
                f"Document {i+1} (Similarity: {similarity:.2f}):\n{metadata['content']}\n"
            )

    if not context_parts:
        context = "No relevant information found."
    else:
        context = "\n".join(context_parts)

    # Step 4: Generate response using an LLM
    # Try to find an LLM plugin
    llm_plugin = None
    for plugin_name in ["ollama", "lmstudio", "openai"]:
        if plugin_name in wdbx.plugins:
            llm_plugin = wdbx.plugins[plugin_name]
            break

    if not llm_plugin:
        raise ValueError("No LLM plugin available")

    # Create prompt with context
    prompt = f"""
Answer the following question based on the provided context:

Question: {query}

Context:
{context}

Answer:"""

    # Generate response
    if hasattr(llm_plugin, "chat"):
        messages = [
            {"role": "system",
             "content":
             "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}]
        response = await llm_plugin.chat(messages, model=model)
    else:
        response = await llm_plugin.generate_text(prompt, model=model)

    return response


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="RAG implementation using WDBX")
    parser.add_argument(
        "--query", default="What is vector search?", help="Question to answer")
    parser.add_argument("--results", type=int, default=5,
                        help="Number of results to retrieve")
    parser.add_argument("--model", default="llama3", help="LLM model to use")
    args = parser.parse_args()

    # Create and initialize WDBX
    wdbx = WDBX(
        vector_dimension=384,
        data_dir="./wdbx_data",
        enable_plugins=True,
    )
    await wdbx.initialize()

    try:
        # Run RAG pipeline
        print(f"Question: {args.query}")
        print("\nGenerating response...")

        response = await rag_pipeline(
            query=args.query,
            wdbx=wdbx,
            num_results=args.results,
            model=args.model
        )

        print("\nResponse:")
        print(response)

    finally:
        # Clean up
        await wdbx.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
