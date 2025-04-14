"""
Ollama integration plugin for WDBX.

This plugin provides integration with Ollama for local LLM and embedding generation.
"""

import os
import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
message, prompt, extra_kwargs):
    parser = argparse.ArgumentParser(
        description="Generate text using a language model")
parser.add_argument("--prompt", required=True,
    type=str, help="Prompt to generate text from")
    parser.add_argument("--model", help="Model to use",
    type=str, default="default",
    parser.add_argument("--temperature", type=float,
    default=0.7, help="Temperature value for generation")
parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens to generate")

    try:
    response = await self.generate(
        prompt=parsed_args.prompt, model=parsed_args.model, temperature=parsed_args.temperature,
    max_tokens=parsed_args.max_tokens,
    print("Generating text...")
    print("Response:")
    await self.generate_text(prompt=parsed_args.prompt,
        model=parsed_args.model,
        temperature=parsed_args.temperature,
        max_tokens=parsed_args.max_tokens)
except Exception as e:
    print(f"Error: {e}")

async def generate_text(self, prompt, model=None,
    system=None, temperature=0.7, max_tokens=50,
    extra_kwargs):
    parser = argparse.ArgumentParser(
        description="Chat with a language model")
parser.add_argument("--model", type=str,
    help="Model to use",
    parser.add_argument("--system", type=str,
    help="System prompt")
parser.add_argument(
    "--temperature", type=float,
    default=0.7, help="Temperature value for generation")
parser.add_argument("--max_tokens", type=int,
    default=50
import numpy as np

from ..plugins.base import WDBXPlugin, PluginError

logger = logging.getLogger(__name__)


class OllamaPlugin(WDBXPlugin):
    """
    Ollama integration plugin for WDBX.

    This plugin allows WDBX to use Ollama for local LLM and embedding generation.

    Attributes:
        wdbx: Reference to the WDBX instance
        host: Ollama API host
        model: Default model to use
        timeout: API request timeout in seconds
    """

    def __init__(self, wdbx):
        """
        Initialize the Ollama plugin.

        Args:
            wdbx: Reference to the WDBX instance
        """
        super().__init__(wdbx)

        # Load configuration
        self.host = self.get_config("HOST", "http://localhost:11434")
        self.model = self.get_config("MODEL", "llama2")
        self.timeout = float(self.get_config("TIMEOUT", 30.0))
        self.embedding_model = self.get_config(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # Initialize session
        self.session = None

        # State tracking
        self.available_models = []
        self.is_connected = False

        logger.info(
            f"Initialized OllamaPlugin with host={self.host}, model={self.model}")

    @property
    def name(self) -> str:
        """Return the name of the plugin."""
        return "ollama"

    @property
    def description(self) -> str:
        """Return a description of the plugin."""
        return "Ollama integration plugin for WDBX, providing local LLM and embedding generation."

    @property
    def version(self) -> str:
        """Return the version of the plugin."""
        return "0.2.0"

    async def initialize(self) -> None:
        """Initialize the plugin."""
        try:
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

            # Check connection and get available models
            try:
                await self._check_connection()
                await self._get_available_models()

                # Pull the default model if it's not available
                if self.model not in self.available_models:
                    await self.pull_model(self.model)

                # Pull the embedding model if it's not available
                if self.embedding_model not in self.available_models:
                    await self.pull_model(self.embedding_model)

                logger.info(f"Connected to Ollama at {self.host}")
                self.is_connected = True
            except Exception as e:
                logger.warning(f"Failed to connect to Ollama: {e}")
                logger.warning("Ollama integration will not be available")

            logger.info(f"OllamaPlugin initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OllamaPlugin: {e}")
            raise PluginError(f"Error initializing OllamaPlugin: {e}")

    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("OllamaPlugin shut down")

    async def _check_connection(self) -> bool:
        """
        Check connection to Ollama API.

        Returns:
            True if connection is successful, False otherwise

        Raises:
            PluginError: If connection check fails
        """
        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to connect to Ollama: {response.status} {response.reason}")
                return True
        except asyncio.TimeoutError:
            raise PluginError(f"Connection to Ollama timed out")
        except Exception as e:
            raise PluginError(f"Failed to connect to Ollama: {e}")

    async def _get_available_models(self) -> List[str]:
        """
        Get list of available models.

        Returns:
            List of available model names

        Raises:
            PluginError: If getting models fails
        """
        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to get available models: {response.status} {response.reason}")

                data = await response.json()
                models = [model["name"] for model in data.get("models", [])]
                self.available_models = models
                return models
        except Exception as e:
            raise PluginError(f"Failed to get available models: {e}")

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama API.

        Args:
            model: Name of the model to pull

        Returns:
            True if the model was pulled successfully

        Raises:
            PluginError: If pulling the model fails
        """
        try:
            logger.info(f"Pulling model: {model}")

            # Send pull request
            async with self.session.post(
                f"{self.host}/api/pull",
                json={"name": model}
            ) as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to pull model {model}: {response.status} {response.reason}")

                # Stream response to get progress updates
                async for line in response.content:
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            raise PluginError(
                                f"Error pulling model: {data['error']}")

                        # Log progress
                        if "completed" in data and data["completed"]:
                            logger.info(f"Model {model} pulled successfully")

                            # Update available models
                            await self._get_available_models()
                            return True
                    except json.JSONDecodeError:
                        pass

                return True
        except Exception as e:
            raise PluginError(f"Failed to pull model {model}: {e}")

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the given text using Ollama.

        Args:
            text: The input text

        Returns:
            Embedding vector as a list of floats

        Raises:
            PluginError: If embedding creation fails
        """
        if not self.is_connected:
            raise PluginError("Ollama plugin not connected")

        try:
            # Ensure the embedding model is available
            if self.embedding_model not in self.available_models:
                await self.pull_model(self.embedding_model)

            # Send embeddings request
            async with self.session.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            ) as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to create embedding: {response.status} {response.reason}")

                data = await response.json()
                embedding = data.get("embedding", [])

                if not embedding:
                    raise PluginError("Empty embedding returned from Ollama")

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
            raise PluginError("Ollama plugin not connected")

        try:
            # Ensure the embedding model is available
            if self.embedding_model not in self.available_models:
                await self.pull_model(self.embedding_model)

            # Create embeddings concurrently
            tasks = []
            for text in texts:
                task = asyncio.create_task(self.create_embedding(text))
                tasks.append(task)

            # Wait for all tasks to complete
            embeddings = await asyncio.gather(*tasks)

            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings batch: {e}")
            raise PluginError(f"Error creating embeddings batch: {e}")

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using Ollama.

        Args:
            prompt: The prompt to generate text from
            model: Model to use (default: plugin's default model)
            system: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text or async generator of text chunks if streaming

        Raises:
            PluginError: If text generation fails
        """
        if not self.is_connected:
            raise PluginError("Ollama plugin not connected")

        # Use default model if not specified
        model = model or self.model

        try:
            # Ensure the model is available
            if model not in self.available_models:
                await self.pull_model(model)

            # Prepare request
            request = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                }
            }

            if system:
                request["system"] = system

            if max_tokens:
                request["options"]["num_predict"] = max_tokens

            # Send generation request
            if stream:
                return self._stream_generation(request)
            else:
                return await self._generate_text(request)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise PluginError(f"Error generating text: {e}")

    async def _generate_text(self, request: Dict[str, Any]) -> str:
        """
        Generate text (non-streaming).

        Args:
            request: Request dictionary

        Returns:
            Generated text

        Raises:
            PluginError: If text generation fails
        """
        try:
            async with self.session.post(
                f"{self.host}/api/generate",
                json=request
            ) as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to generate text: {response.status} {response.reason}")

                data = await response.json()
                return data.get("response", "")
        except Exception as e:
            raise PluginError(f"Failed to generate text: {e}")

    async def _stream_generation(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream text generation.

        Args:
            request: Request dictionary

        Yields:
            Text chunks as they are generated

        Raises:
            PluginError: If text generation fails
        """
        try:
            async with self.session.post(
                f"{self.host}/api/generate",
                json=request
            ) as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to generate text: {response.status} {response.reason}")

                # Stream response
                async for line in response.content:
                    try:
                        data = json.loads(line)
                        if "error" in data:
                            raise PluginError(
                                f"Error generating text: {data['error']}")

                        if "response" in data:
                            yield data["response"]

                        # Check if generation is done
                        if "done" in data and data["done"]:
                            break
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            raise PluginError(f"Failed to stream text generation: {e}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Chat with Ollama.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model to use (default: plugin's default model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated response or async generator of response chunks if streaming

        Raises:
            PluginError: If chat fails
        """
        if not self.is_connected:
            raise PluginError("Ollama plugin not connected")

        # Use default model if not specified
        model = model or self.model

        try:
            # Ensure the model is available
            if model not in self.available_models:
                await self.pull_model(model)

            # Extract system message
            system = None
            chat_messages = []

            for message in messages:
                if message["role"] == "system":
                    system = message["content"]
                else:
                    chat_messages.append(message)

            # Format chat messages
            chat_string = ""
            for message in chat_messages:
                if message["role"] == "user":
                    chat_string += f"[INST] {message['content']} [/INST]\n"
                else:
                    chat_string += f"{message['content']}\n"

            # Generate text
            return await self.generate_text(
                prompt=chat_string,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise PluginError(f"Error in chat: {e}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.

        Returns:
            List of model information dictionaries

        Raises:
            PluginError: If listing models fails
        """
        if not self.is_connected:
            raise PluginError("Ollama plugin not connected")

        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to list models: {response.status} {response.reason}")

                data = await response.json()
                return data.get("models", [])
        except Exception as e:
            raise PluginError(f"Failed to list models: {e}")

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Name of the model

        Returns:
            Model information dictionary

        Raises:
            PluginError: If getting model info fails
        """
        if not self.is_connected:
            raise PluginError("Ollama plugin not connected")

        try:
            async with self.session.post(
                f"{self.host}/api/show",
                json={"name": model}
            ) as response:
                if response.status != 200:
                    raise PluginError(
                        f"Failed to get model info: {response.status} {response.reason}")

                return await response.json()
        except Exception as e:
            raise PluginError(f"Failed to get model info: {e}")

    def register_commands(self) -> None:
        """Register commands with the WDBX CLI."""
        if hasattr(self.wdbx, "register_command"):
            self.wdbx.register_command(
                "ollama-generate",
                self._cmd_generate,
                "Generate text using Ollama",
                {
                    "--prompt": "Prompt to generate text from",
                    "--model": "Model to use",
                    "--system": "System prompt",
                    "--temperature": "Sampling temperature (0-1)",
                    "--max-tokens": "Maximum number of tokens to generate",
                }
            )

            self.wdbx.register_command(
                "ollama-chat",
                self._cmd_chat,
                "Chat with Ollama",
                {
                    "--message": "User message",
                    "--model": "Model to use",
                    "--system": "System prompt",
                    "--temperature": "Sampling temperature (0-1)",
                    "--max-tokens": "Maximum number of tokens to generate",
                }
            )

            self.wdbx.register_command(
                "ollama-models",
                self._cmd_models,
                "List available Ollama models",
                {}
            )

            self.wdbx.register_command(
                "ollama-pull",
                self._cmd_pull,
                "Pull a model from Ollama",
                {
                    "--model": "Model to pull",
                }
            )

    async def _cmd_generate(self, args):
        """Command handler for the ollama-generate command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Generate text using Ollama")
        parser.add_argument("--prompt", required=True,
                            help="Prompt to generate text from")
        parser.add_argument("--model", help="Model to use")
        parser.add_argument("--system", help="System prompt")
        parser.add_argument(
            "--temperature", type=float, default=0.7,
            help="Sampling temperature (0-1)")
        parser.add_argument(
            "--max-tokens", type=int,
            help="Maximum number of tokens to generate")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Generate text
            response = await self.generate_text(
                prompt=parsed_args.prompt,
                model=parsed_args.model,
                system=parsed_args.system,
                temperature=parsed_args.temperature,
                max_tokens=parsed_args.max_tokens
            )

            # Print response
            print("Generated text:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_chat(self, args):
        """Command handler for the ollama-chat command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(description="Chat with Ollama")
        parser.add_argument("--message", required=True, help="User message")
        parser.add_argument("--model", help="Model to use")
        parser.add_argument("--system", help="System prompt")
        parser.add_argument(
            "--temperature", type=float, default=0.7,
            help="Sampling temperature (0-1)")
        parser.add_argument(
            "--max-tokens", type=int,
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

            # Chat
            response = await self.chat(
                messages=messages,
                model=parsed_args.model,
                temperature=parsed_args.temperature,
                max_tokens=parsed_args.max_tokens
            )

            # Print response
            print("Response:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_models(self, args):
        """Command handler for the ollama-models command."""
        try:
            # List models
            models = await self.list_models()

            # Print models
            print("Available models:")
            for model in models:
                print(f"  {model['name']} ({model.get('size', 'unknown')})")
        except Exception as e:
            print(f"Error: {e}")

    async def _cmd_pull(self, args):
        """Command handler for the ollama-pull command."""
        import argparse

        # Parse arguments
        parser = argparse.ArgumentParser(
            description="Pull a model from Ollama")
        parser.add_argument("--model", required=True, help="Model to pull")

        try:
            parsed_args = parser.parse_args(args.split())
        except SystemExit:
            return

        try:
            # Pull model
            print(f"Pulling model: {parsed_args.model}")
            await self.pull_model(parsed_args.model)
            print(f"Model {parsed_args.model} pulled successfully")
        except Exception as e:
            print(f"Error: {e}")
