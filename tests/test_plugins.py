"""
Unit tests for WDBX plugins.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from wdbx import WDBX
from wdbx.plugins.base import WDBXPlugin, PluginError
from wdbx.plugins.webscraper import WebScraperPlugin
from wdbx.plugins.ollama import OllamaPlugin
from wdbx.plugins.lmstudio import LMStudioPlugin


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def wdbx_with_plugins(temp_dir):
    """Create and initialize a WDBX instance with plugins enabled."""
    wdbx = WDBX(
        vector_dimension=384,
        num_shards=1,
        data_dir=temp_dir,
        enable_plugins=True,
    )

    # Initialize
    await wdbx.initialize()

    yield wdbx

    # Clean up
    await wdbx.shutdown()

# Test loading plugins


@pytest.mark.asyncio
async def test_plugins_loaded(wdbx_with_plugins):
    """Test that plugins are loaded correctly."""
    # Check that some plugins are loaded
    assert len(wdbx_with_plugins.plugins) > 0

    # Get plugin info
    plugin_names = [plugin.name for plugin in
                    wdbx_with_plugins.plugins.values()]
    assert "webscraper" in plugin_names


@pytest.mark.asyncio
async def test_plugin_registration():
    """Test plugin registration."""
    # Create WDBX instance
    wdbx = WDBX(vector_dimension=4, enable_plugins=False)

    # Create a custom plugin
    class TestPlugin(WDBXPlugin):
        @property
        def name(self):
            return "test_plugin"

        @property
        def description(self):
            return "Test plugin"

        @property
        def version(self):
            return "0.1.0"

    # Register the plugin
    plugin = TestPlugin(wdbx)
    result = wdbx.register_plugin(plugin)

    # Verify plugin was registered
    assert result is True
    assert "test_plugin" in wdbx.plugins
    assert wdbx.plugins["test_plugin"] is plugin

    # Clean up
    await wdbx.shutdown()

# Test WebScraperPlugin


@pytest.mark.asyncio
async def test_webscraper_plugin(wdbx_with_plugins):
    """Test WebScraperPlugin functionality."""
    # Skip if webscraper plugin is not available
    if "webscraper" not in wdbx_with_plugins.plugins:
        pytest.skip("WebScraperPlugin not available")

    webscraper = wdbx_with_plugins.plugins["webscraper"]

    # Mock aiohttp.ClientSession for extract_content
    with patch("aiohttp.ClientSession") as mock_session:
        # Configure the mock
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text.return_value = "<html><body><h1>Test Page</h1><p>This is a test.</p></body></html>"

        # Configure the session
        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__.return_value = mock_response
        mock_session.return_value = MagicMock()
        mock_session.return_value.get.return_value = mock_session_instance

        # Set the mock session
        webscraper.session = mock_session.return_value

        # Test extract_content
        content = await webscraper.extract_content("https://example.com")

        # Verify content
        assert "Test Page" in content
        assert "This is a test." in content


@pytest.mark.asyncio
async def test_webscraper_embedding(wdbx_with_plugins):
    """Test WebScraperPlugin embedding creation."""
    # Skip if webscraper plugin is not available
    if "webscraper" not in wdbx_with_plugins.plugins:
        pytest.skip("WebScraperPlugin not available")

    webscraper = wdbx_with_plugins.plugins["webscraper"]

    # Mock embedding model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.randn(384)
    webscraper.embedding_model = mock_model

    # Test create_embedding
    embedding = await webscraper.create_embedding("Test text")

    # Verify embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 384

    # Verify model was called
    mock_model.encode.assert_called_once_with("Test text")

# Test OllamaPlugin


@pytest.mark.asyncio
async def test_ollama_plugin():
    """Test OllamaPlugin functionality."""
    # Create WDBX instance without initializing plugins
    wdbx = WDBX(vector_dimension=384, enable_plugins=False)

    # Create OllamaPlugin
    ollama = OllamaPlugin(wdbx)

    # Mock aiohttp.ClientSession for API calls
    with patch("aiohttp.ClientSession") as mock_session:
        # Configure the mock
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * 384}
            ]
        }

        # Configure the session
        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__.return_value = mock_response
        mock_session.return_value = MagicMock()
        mock_session.return_value.post.return_value = mock_session_instance

        # Set the mock session and state
        ollama.session = mock_session.return_value
        ollama.is_connected = True
        ollama.embedding_model = "test-model"

        # Test create_embedding
        embedding = await ollama.create_embedding("Test text")

        # Verify embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(x == 0.1 for x in embedding)

        # Verify API call
        mock_session.return_value.post.assert_called_once()
        call_args = mock_session.return_value.post.call_args[0][0]
        assert "embeddings" in call_args

    # Clean up
    await wdbx.shutdown()

# Test LMStudioPlugin


@pytest.mark.asyncio
async def test_lmstudio_plugin():
    """Test LMStudioPlugin functionality."""
    # Create WDBX instance without initializing plugins
    wdbx = WDBX(vector_dimension=384, enable_plugins=False)

    # Create LMStudioPlugin
    lmstudio = LMStudioPlugin(wdbx)

    # Mock aiohttp.ClientSession for API calls
    with patch("aiohttp.ClientSession") as mock_session:
        # Configure the mock for embeddings
        mock_embedding_response = MagicMock()
        mock_embedding_response.status = 200
        mock_embedding_response.json.return_value = {
            "data": [
                {"embedding": [0.2] * 384}
            ]
        }

        # Configure the mock for chat
        mock_chat_response = MagicMock()
        mock_chat_response.status = 200
        mock_chat_response.json.return_value = {
            "choices": [
                {"message": {"content": "Test response"}}
            ]
        }

        # Configure the session
        mock_session.return_value = MagicMock()

        # Setup response mapping
        def get_response(url, **kwargs):
            if "embeddings" in url:
                mock_resp = mock_embedding_response
            else:
                mock_resp = mock_chat_response

            mock_resp_context = MagicMock()
            mock_resp_context.__aenter__.return_value = mock_resp
            return mock_resp_context

        mock_session.return_value.post.side_effect = get_response

        # Set the mock session and state
        lmstudio.session = mock_session.return_value
        lmstudio.is_connected = True
        lmstudio.model = "test-model"
        lmstudio.embedding_model = "test-model"

        # Test create_embedding
        embedding = await lmstudio.create_embedding("Test text")

        # Verify embedding
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(x == 0.2 for x in embedding)

        # Test chat
        messages = [
            {"role": "user", "content": "Test message"}
        ]
        response = await lmstudio.chat(messages)

        # Verify response
        assert response == "Test response"

    # Clean up
    await wdbx.shutdown()

# Test Plugin Manager


@pytest.mark.asyncio
async def test_plugin_manager():
    """Test PluginManager functionality."""
    from wdbx.plugins.base import PluginManager

    # Create WDBX instance
    wdbx = WDBX(vector_dimension=4, enable_plugins=False)

    # Create plugin manager
    manager = PluginManager(wdbx)

    # Create a custom plugin
    class TestPlugin(WDBXPlugin):
        @property
        def name(self):
            return "test_plugin"

        @property
        def description(self):
            return "Test plugin"

        @property
        def version(self):
            return "0.1.0"

    # Register the plugin
    plugin = TestPlugin(wdbx)
    result = manager.register_plugin(plugin)

    # Verify plugin was registered
    assert result is True
    assert "test_plugin" in manager.plugins

    # Get plugin info
    plugin_info = manager.get_plugin_info()
    assert len(plugin_info) == 1
    assert plugin_info[0]["name"] == "test_plugin"
    assert plugin_info[0]["version"] == "0.1.0"

    # Get plugin by name
    retrieved_plugin = manager.get_plugin("test_plugin")
    assert retrieved_plugin is plugin

    # Get nonexistent plugin
    nonexistent = manager.get_plugin("nonexistent")
    assert nonexistent is None

    # Get stats
    stats = manager.get_stats()
    assert stats["num_plugins"] == 1

    # Unregister plugin
    result = manager.unregister_plugin("test_plugin")
    assert result is True
    assert "test_plugin" not in manager.plugins

    # Clean up
    await wdbx.shutdown()

# Test Plugin Errors


@pytest.mark.asyncio
async def test_plugin_errors():
    """Test plugin error handling."""
    # Create WDBX instance
    wdbx = WDBX(vector_dimension=4, enable_plugins=False)

    # Create a custom plugin
    class ErrorPlugin(WDBXPlugin):
        @property
        def name(self):
            return "error_plugin"

        @property
        def description(self):
            return "Error plugin"

        @property
        def version(self):
            return "0.1.0"

        async def initialize(self):
            raise RuntimeError("Test initialization error")

        async def create_embedding(self, text):
            raise PluginError("Test embedding error")

    # Register the plugin
    plugin = ErrorPlugin(wdbx)
    wdbx.register_plugin(plugin)

    # Test initialization error
    with pytest.raises(RuntimeError, match="Test initialization error"):
        await plugin.initialize()

    # Test embedding error
    with pytest.raises(PluginError, match="Test embedding error"):
        await plugin.create_embedding("Test")

    # Clean up
    await wdbx.shutdown()
