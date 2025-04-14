"""
WDBX: Vector Database for AI Applications
=========================================

A flexible vector database system designed for AI applications with an extensible plugin architecture.

Features:
- High-performance vector storage and similarity search
- Extensible plugin architecture for easy integration with external services
- Built-in support for various embedding models and LLM providers
- Asynchronous API for non-blocking operations
- Visualization and analytics capabilities
- Secure storage with support for encryption
"""

__version__ = "0.2.0"
__author__ = "WDBX Team"

from .core.wdbx import WDBX
from .core.config import WDBXConfig
from .plugins.base import WDBXPlugin, PluginError

__all__ = ["WDBX", "WDBXConfig", "WDBXPlugin", "PluginError"]
