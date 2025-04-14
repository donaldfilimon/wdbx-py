"""
RESTful API server for WDBX.

This module provides a FastAPI-based server for accessing WDBX functionality remotely.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class WDBXAPIServer:
    """
    API server for WDBX.

    This class provides a FastAPI-based server for accessing WDBX functionality remotely.

    Attributes:
        wdbx: Reference to the WDBX instance
        host: Host to bind the server to
        port: Port to listen on
        enable_auth: Whether to enable authentication
        auth_key: Authentication key
        app: FastAPI application
    """

    def __init__(
        self,
        wdbx,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_auth: bool = False,
        auth_key: Optional[str] = None,
        enable_cors: bool = True,
        cors_origins: Optional[List[str]] = None,
    ):
        """
        Initialize the API server.

        Args:
            wdbx: Reference to the WDBX instance
            host: Host to bind the server to
            port: Port to listen on
            enable_auth: Whether to enable authentication
            auth_key: Authentication key
            enable_cors: Whether to enable CORS
            cors_origins: List of allowed CORS origins
        """
        self.wdbx = wdbx
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.auth_key = auth_key
        self.enable_cors = enable_cors
        self.cors_origins = cors_origins or ["*"]

        # Initialize FastAPI
        self.app = None
        self.api_router = None

        logger.info(f"Initialized WDBXAPIServer on {host}:{port}")

    async def initialize(self):
        """Initialize the API server."""
        try:
            from fastapi import FastAPI, Depends, HTTPException, status, Security, APIRouter
            from fastapi.security.api_key import APIKeyHeader
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel, Field

            # Create FastAPI application
            self.app = FastAPI(
                title="WDBX API",
                description="API for WDBX vector database",
                version=self.wdbx.version,
            )

            # Create API router
            self.api_router = APIRouter()

            # Set up CORS
            if self.enable_cors:
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=self.cors_origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )

            # Set up authentication
            if self.enable_auth:
                api_key_header = APIKeyHeader(name="X-API-Key")

                async def get_api_key(api_key: str = Security(api_key_header)):
                    if api_key != self.auth_key:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API Key",
                        )
                    return api_key

                # Apply authentication to all routes
                self.api_router = APIRouter(
                    dependencies=[Depends(get_api_key)])

            # Define data models
            class VectorModel(BaseModel):
                vector: List[float]
                metadata: Optional[Dict[str, Any]] = None
                id: Optional[str] = None

            class SearchModel(BaseModel):
                query_vector: List[float]
                limit: Optional[int] = 10
                threshold: Optional[float] = 0.0
                filter_metadata: Optional[Dict[str, Any]] = None

            class MetadataModel(BaseModel):
                metadata: Dict[str, Any]

            class EmbeddingModel(BaseModel):
                text: str

            class TextsModel(BaseModel):
                texts: List[str]

            # Define routes

            # Health check
            @self.api_router.get("/health")
            async def health_check():
                return {"status": "healthy", "version": self.wdbx.version}

            # Vector operations
            @self.api_router.post("/vectors")
            async def store_vector(vector_data: VectorModel):
                vector_id = await self.wdbx.vector_store_async(
                    vector_data.vector,
                    vector_data.metadata,
                    vector_data.id
                )
                return {"vector_id": vector_id}

            @self.api_router.post("/vectors/search")
            async def search_vectors(search_data: SearchModel):
                results = await self.wdbx.vector_search_async(
                    search_data.query_vector,
                    search_data.limit,
                    search_data.threshold,
                    search_data.filter_metadata
                )
                return {"results": [
                    {"vector_id": vid, "similarity": sim, "metadata": meta}
                    for vid, sim, meta in results
                ]}

            @self.api_router.get("/vectors/{vector_id}")
            async def get_vector(vector_id: str):
                result = await self.wdbx.get_vector_async(vector_id)
                if result is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Vector with ID {vector_id} not found",
                    )
                vector, metadata = result
                return {"vector_id": vector_id, "vector": vector, "metadata": metadata}

            @self.api_router.delete("/vectors/{vector_id}")
            async def delete_vector(vector_id: str):
                success = await self.wdbx.delete_vector_async(vector_id)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Vector with ID {vector_id} not found",
                    )
                return {"success": True}

            @self.api_router.put("/vectors/{vector_id}/metadata")
            async def update_metadata(
                    vector_id: str, metadata_data: MetadataModel):
                success = await self.wdbx.update_metadata_async(vector_id, metadata_data.metadata)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Vector with ID {vector_id} not found",
                    )
                return {"success": True}

            # Database operations
            @self.api_router.get("/stats")
            async def get_stats():
                return self.wdbx.get_stats()

            @self.api_router.post("/clear")
            async def clear_database():
                count = await self.wdbx.clear_async()
                return {"removed_vectors": count}

            # Embedding operations
            @self.api_router.post("/embeddings")
            async def create_embedding(data: EmbeddingModel):
                # Try to find a plugin that can generate embeddings
                for plugin_name in [
                        "openai", "ollama", "huggingface", "sentencetransformers"]:
                    if plugin_name in self.wdbx.plugins:
                        plugin = self.wdbx.plugins[plugin_name]
                        try:
                            embedding = await plugin.create_embedding(data.text)
                            return {"embedding": embedding}
                        except Exception as e:
                            logger.error(
                                f"Error creating embedding with {plugin_name}: {e}")

                # If no plugin is available, return an error
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="No embedding plugin available",
                )

            @self.api_router.post("/embeddings/batch")
            async def create_embeddings_batch(data: TextsModel):
                # Try to find a plugin that can generate embeddings
                for plugin_name in [
                        "openai", "ollama", "huggingface", "sentencetransformers"]:
                    if plugin_name in self.wdbx.plugins:
                        plugin = self.wdbx.plugins[plugin_name]
                        try:
                            if hasattr(plugin, "create_embeddings_batch"):
                                embeddings = await plugin.create_embeddings_batch(data.texts)
                            else:
                                # Fall back to individual embeddings
                                embeddings = []
                                for text in data.texts:
                                    embedding = await plugin.create_embedding(text)
                                    embeddings.append(embedding)

                            return {"embeddings": embeddings}
                        except Exception as e:
                            logger.error(
                                f"Error creating embeddings batch with {plugin_name}: {e}")

                # If no plugin is available, return an error
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="No embedding plugin available",
                )

            # Plugin operations
            @self.api_router.get("/plugins")
            async def list_plugins():
                return {
                    "plugins": [
                        {
                            "name": plugin.name,
                            "description": plugin.description,
                            "version": plugin.version,
                        }
                        for plugin in self.wdbx.plugins.values()
                    ]
                }

            @self.api_router.get("/plugins/{plugin_name}")
            async def get_plugin_info(plugin_name: str):
                if plugin_name not in self.wdbx.plugins:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Plugin {plugin_name} not found",
                    )

                plugin = self.wdbx.plugins[plugin_name]
                return {
                    "name": plugin.name,
                    "description": plugin.description,
                    "version": plugin.version,
                    "stats": plugin.get_stats(),
                }

            # Include the API router
            self.app.include_router(self.api_router, prefix="/api/v1")

            logger.info("API server initialized")
        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            logger.error(
                "FastAPI is required for the API server. Install with: pip install fastapi uvicorn")
            raise RuntimeError(f"Required package not installed: {e}")
        except Exception as e:
            logger.error(f"Error initializing API server: {e}")
            raise RuntimeError(f"Error initializing API server: {e}")

    async def start(self):
        """Start the API server."""
        try:
            import uvicorn

            # Initialize if not already initialized
            if not self.app:
                await self.initialize()

            # Create uvicorn config
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
            )

            # Create and start server
            server = uvicorn.Server(config)
            await server.serve()

            logger.info(f"API server started on {self.host}:{self.port}")
        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            logger.error(
                "Uvicorn is required for the API server. Install with: pip install uvicorn")
            raise RuntimeError(f"Required package not installed: {e}")
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            raise RuntimeError(f"Error starting API server: {e}")

    def start_in_thread(self):
        """Start the API server in a separate thread."""
        import threading
        import asyncio

        # Create new event loop for the thread
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())

        # Start server in thread
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

        logger.info(f"API server started in thread on {self.host}:{self.port}")
        return thread
