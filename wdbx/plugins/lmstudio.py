"""
LMStudio integration plugin for WDBX.

This plugin provides integration with LMStudio's OpenAI-compatible API
for local LLM and embedding generation.
                                            async_ await        pass #from
 import unused # np as numpy import#
Tuple loggingimportsavesave    #>this_rewrite"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
import aiohttp
import numpy as np

from ..plugins.base import WDBXPlugin, PluginError

logger = logging.getLogger(__name__)


class LMStudioPlugin(WDBXPlugin):
    """
    LMStudio integration plugin for WDBX.

    This plugin allows WDBX to use LMStudio's OpenAI-compatible API for
    local LLM and embedding generation.

    Attributes:
        wdbx: Reference to the WDBX instance
        host: LMStudio API host
        port: LMStudio API port
        api_key: API key (usually not required for local instances)
        timeout: Request timeout in seconds
        is_connected: Whether the plugin is connected to LMStudio
    """

    def __init__(self, wdbx):
        """
        Initialize the LMStudio plugin.

        Args:
            wdbx: Reference to the WDBX instance
        """
        super().__init__(wdbx)

        # Load configuration
        self.host = self.get_config("HOST", "localhost")
        self.port = int(self.get_config("PORT", 8000))
        self.api_key = self.get_config("API_KEY", "")
        self.timeout = float(self.get_config("TIMEOUT", 30.0))
        self.model = self.get_config("MODEL", "")
        self.embedding_model = self.get_config("EMBEDDING_MODEL", "")

        # Build base URL
        self.base_url = f"http://{self.host}:{self.port}/v1"

        # Initialize session
        self.session = None

        # State tracking
        self.is_connected = False
        self.available_models = []

        logger.info(
            f"Initialized LMStudioPlugin with base_url={self.base_url}")

    @property
    def name(self) -> str:
        """Return the name of the plugin."""
        return "lmstudio"

    @property
    def description(self) -> str:
        """Return a description of the plugin."""
        return "LMStudio integration plugin for WDBX, providing access to local LLMs through OpenAI-compatible API."

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "0.2.0"

    async def initialize(self) -> None:
        """Initialize the plugin."""
        try:
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self._get_headers()
            )

            # Check connection
            try:
                await self._check_connection()
                await self._get_available_models()
                self.is_connected = True
                logger.info(f"Connected to LMStudio at {self.base_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to LMStudio: {e}")
                logger.warning("LMStudio integration will not be available")

            logger.info(f"LMStudioPlugin initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LMStudioPlugin: {e}")
            raise PluginError(f"Error initializing LMStudioPlugin: {e}")

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("LMStudioPlugin shut down")

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.

        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def _check_connection(self) -> bool:
        """
        Check connection to LMStudio API.

        Returns:
            True if connection is successful, False otherwise

        Raises:
            PluginError: If connection check fails
        """
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to connect to LMStudio: {response.status} {error_text}")
                return True
        except asyncio.TimeoutError:
            raise PluginError(f"Connection to LMStudio timed out")
        except Exception as e:
            raise PluginError(f"Failed to connect to LMStudio: {e}")

    async def _get_available_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List of available model names

        Raises:
            PluginError: If getting models fails
        """
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to get available models: {response.status} {error_text}")

                data = await response.json()
                models = [model["id"] for model in data["data"]]
                self.available_models = models

                # Set default model if not specified
                if not self.model and models:
                    self.model = models[0]

                # Set default embedding model if not specified
                if not self.embedding_model and models:
                    self.embedding_model = models[0]

                return models
        except Exception as e:
            raise PluginError(f"Failed to get available models: {e}")

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text using LMStudio.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats

        Raises:
            PluginError: If embedding creation fails
        """
        if not self.is_connected:
            raise PluginError("LMStudio plugin not connected")

        try:
            # Ensure we have a model
            if not self.embedding_model:
                if not self.available_models:
                    await self._get_available_models()
                if not self.embedding_model and self.available_models:
                    self.embedding_model = self.available_models[0]

            if not self.embedding_model:
                raise PluginError("No embedding model available")

            # Prepare request
            data = {
                "input": text,
                "model": self.embedding_model
            }

            # Send request
            async with self.session.post(f"{self.base_url}/embeddings", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to create embedding: {response.status} {error_text}")

                result = await response.json()
                embedding = result["data"][0]["embedding"]

                return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise PluginError(f"Error creating embedding: {e}")

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embedding vectors

        Raises:
            PluginError: If embedding creation fails
        """
        if not self.is_connected:
            raise PluginError("LMStudio plugin not connected")

        try:
            # Ensure we have a model
            if not self.embedding_model:
                if not self.available_models:
                    await self._get_available_models()
                if not self.embedding_model and self.available_models:
                    self.embedding_model = self.available_models[0]

            if not self.embedding_model:
                raise PluginError("No embedding model available")

            # Prepare request
            data = {
                "input": texts,
                "model": self.embedding_model
            }

            # Send request
            async with self.session.post(f"{self.base_url}/embeddings", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to create embeddings: {response.status} {error_text}")

                result = await response.json()
                embeddings = [item["embedding"] for item in result["data"]]

                return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings batch: {e}")
            raise PluginError(f"Error creating embeddings batch: {e}")

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False,
        stop: Optional[List[str]] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text completion using LMStudio.

        Args:
            prompt: The prompt to complete
            model: Model to use (default: plugin's default model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            stop: Optional list of stop sequences

        Returns:
            Generated text or async generator of text chunks if streaming

        Raises:
            PluginError: If text generation fails
        """
        if not self.is_connected:
            raise PluginError("LMStudio plugin not connected")

        # Use default model if not specified
        model = model or self.model

        # Ensure we have a model
        if not model:
            if not self.available_models:
                await self._get_available_models()
            if not model and self.available_models:
                model = self.available_models[0]

        if not model:
            raise PluginError("No model available")

        try:
            # Prepare request
            data = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            if stop:
                data["stop"] = stop

            # Send request
            if stream:
                return self._stream_completion(data)
            else:
                return await self._complete(data)
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise PluginError(f"Error generating completion: {e}")

    async def _complete(self, data: Dict[str, Any]) -> str:
        """
        Generate text completion (non-streaming).

        Args:
            data: Request data

        Returns:
            Generated text

        Raises:
            PluginError: If text generation fails
        """
        try:
            async with self.session.post(f"{self.base_url}/completions", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to generate completion: {response.status} {error_text}")

                result = await response.json()
                return result["choices"][0]["text"]
        except Exception as e:
            raise PluginError(f"Failed to generate completion: {e}")

    async def _stream_completion(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream text completion.

        Args:
            data: Request data

        Yields:
            Text chunks as they are generated

        Raises:
            PluginError: If text generation fails
        """
        try:
            async with self.session.post(f"{self.base_url}/completions", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to generate completion: {response.status} {error_text}")

                # Stream response
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        if line.startswith(b"data: "):
                            line = line[6:]  # Remove "data: " prefix

                        # Check for end of stream
                        if line == b"[DONE]":
                            break

                        # Parse JSON
                        data = json.loads(line)

                        # Extract text
                        if "choices" in data and data["choices"] and "text" in data["choices"][0]:
                            yield data["choices"][0]["text"]
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            raise PluginError(f"Failed to stream completion: {e}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False,
        stop: Optional[List[str]] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Chat completion using LMStudio.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (default: plugin's default model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            stop: Optional list of stop sequences

        Returns:
            Generated response or async generator of response chunks if streaming

        Raises:
            PluginError: If chat completion fails
        """
        if not self.is_connected:
            raise PluginError("LMStudio plugin not connected")

        # Use default model if not specified
        model = model or self.model

        # Ensure we have a model
        if not model:
            if not self.available_models:
                await self._get_available_models()
            if not model and self.available_models:
                model = self.available_models[0]

        if not model:
            raise PluginError("No model available")

        try:
            # Prepare request
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            if stop:
                data["stop"] = stop

            # Send request
            if stream:
                return self._stream_chat(data)
            else:
                return await self._chat(data)
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            raise PluginError(f"Error generating chat completion: {e}")

    async def _chat(self, data: Dict[str, Any]) -> str:
        """
        Generate chat completion (non-streaming).

        Args:
            data: Request data

        Returns:
            Generated response

        Raises:
            PluginError: If chat completion fails
        """
        try:
            async with self.session.post(f"{self.base_url}/chat/completions", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to generate chat completion: {response.status} {error_text}")

                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise PluginError(f"Failed to generate chat completion: {e}")

    async def _stream_chat(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream chat completion.

        Args:
            data: Request data

        Yields:
            Response chunks as they are generated

        Raises:
            PluginError: If chat completion fails
        """
        try:
            async with self.session.post(f"{self.base_url}/chat/completions", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise PluginError(
                        f"Failed to generate chat completion: {response.status} {error_text}")

                # Stream response
                async for line in response.content:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        if line.startswith(b"data: "):
                            line = line[6:]  # Remove "data: " prefix

                        # Check for end of stream
                        if line == b"[DONE]":
                            break

                        # Parse JSON
                        data = json.loads(line)

                        # Extract content
                        if "choices" in data and data["choices"] and "delta" in data["choices"][0]:
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            raise PluginError(f"Failed to stream chat completion: {e}")

    def register_commands(self) -> None:
        """Register commands with the WDBX CLI."""
        if hasattr(self.wdbx, "register_command"):
            self.wdbx.register_command(
                "lmstudio-chat",
                self._cmd_chat,
                "Chat with LMStudio model",
                {
                    "--message": "User message",
                    "--model": "Model to use",
                    "--system": "System prompt",
                    "--temperature": "Sampling temperature (0-1)",
                    "--max-tokens": "Maximum number of tokens to generate",
                }
            )

            self.wdbx.register_command(
                "lmstudio-complete",
                self._cmd_complete,
                "Generate text completion with LMStudio model",
                {
                    "--prompt": "Completion prompt",
                    "--model": "Model to use",
                    "--temperature": "Sampling temperature (0-1)",
                    "--max-tokens": "Maximum number of tokens to generate",
                }
            )

            self.wdbx.register_command(
                "lmstudio-models",
                self._cmd_models,
                "List available LMStudio models",
                {}
            )

    async def _cmd_chat(self, args: str):
        """Command handler for the lmstudio-chat command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Chat with LMStudio model")
        parser.add_argument("--message", required=True, help="User message")
        parser.add_argument("--model", help="Model to use")
        parser.add_argument("--system", help="System prompt")
        parser.add_argument(
            "--temperature", type=float, default=0.7,
            help="Sampling temperature (0-1)")
        parser.add_argument(
            "--max-tokens", type=int, default=256,
            help="Maximum number of tokens to generate")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Prepare messages
            messages = []

            if parsed_args.system:
                messages.append(
                    {"role": "system", "content": parsed_args.system})

            messages.append({"role": "user", "content": parsed_args.message})

            # Generate response
            print("Generating response...")
            response = await self.chat(
                messages=messages,
                model=parsed_args.model,
                temperature=parsed_args.temperature,
                max_tokens=parsed_args.max_tokens
            )

            # Print response
            print("\nResponse:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_complete(self, args: str):
        """Command handler for the lmstudio-complete command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Generate text completion with LMStudio model")
        parser.add_argument("--prompt", required=True,
                            help="Completion prompt")
        parser.add_argument("--model", help="Model to use")
        parser.add_argument(
            "--temperature", type=float, default=0.7,
            help="Sampling temperature (0-1)")
        parser.add_argument(
            "--max-tokens", type=int, default=256,
            help="Maximum number of tokens to generate")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Generate completion
            print("Generating completion...")
            completion = await self.complete(
                prompt=parsed_args.prompt,
                model=parsed_args.model,
                temperature=parsed_args.temperature,
                max_tokens=parsed_args.max_tokens
            )

            # Print completion
            print("\nCompletion:")
            print(completion)
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_models(self, args: str):
        """Command handler for the lmstudio-models command."""
        try:
            # Get available models
            models = await self._get_available_models()

            # Print models
            print("Available models:")
            for model in models:
                print(f"  {model}")

            # Print current model
            if self.model:
                print(f"\nCurrent model: {self.model}")

            # Print current embedding model
            if self.embedding_model:
                print(f"Current embedding model: {self.embedding_model}")
        except Exception as e:
            print(f"Error: {e}")
