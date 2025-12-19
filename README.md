# WDBX: Vector Database for AI Applications

[![PyPI version](https://img.shields.io/pypi/v/wdbx.svg)](https://pypi.org/project/wdbx/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wdbx.svg)](https://pypi.org/project/wdbx/)
[![License](https://img.shields.io/pypi/l/wdbx.svg)](https://github.com/wdbx/wdbx_python/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

WDBX is a flexible vector database system designed for AI applications with an extensible plugin architecture.

## Features

- 🚀 High-performance vector storage and similarity search with multiple indexing options
- 🔄 Asynchronous API for non-blocking operations
- 🔌 Extensible plugin architecture for easy integration with external services
- 🌐 RESTful API server for remote access
- 🤖 Built-in support for various embedding models and LLM providers
- 📊 Advanced visualization and analytics capabilities
- 🔄 Distributed architecture with sharding and replication
- 🔒 Secure storage with support for authentication and encryption
- 💻 Command-line interface for easy management

## Installation

```bash
pip install wdbx
```

To install with specific components:

```bash
pip install wdbx[api]          # Install with API server
pip install wdbx[security]     # Install with security features
pip install wdbx[visualization] # Install with visualization tools
pip install wdbx[indexing]     # Install with advanced indexing
pip install wdbx[webscraper]   # Install with web scraper plugin
pip install wdbx[ollama]       # Install with Ollama integration
pip install wdbx[all]          # Install with all components
```

### Docker Installation

To run WDBX using Docker, you can use the provided `docker-compose.yml` file:

```bash
docker-compose up -d
```

This will start the WDBX API server and other services defined in the `docker-compose.yml` file.

## Configuration

WDBX can be configured using a YAML configuration file located at `config/wdbx_config.yaml`. Below are the available configuration options:

```yaml
# WDBX Configuration

# Core settings
vector_dimension: 384
num_shards: 2
data_dir: "./wdbx_data"
enable_plugins: true
enable_distributed: false
enable_gpu: false
log_level: "INFO"

# Vector storage settings
vector_store:
  save_immediately: false
  threads: 4
  cache_size_mb: 128

# Index settings
indexing:
  type: "hnsw" # "hnsw" or "faiss"
  hnsw:
    m: 16
    ef_construction: 200
    ef_search: 50
  faiss:
    index_type: "Flat"
    nprobe: 8

# API server settings
api:
  host: "0.0.0.0"
  port: 8000
  enable_auth: false
  auth_key: ""
  enable_cors: true
  cors_origins: ["*"]

# Plugin settings
plugins:
  # WebScraper plugin
  webscraper:
    user_agent: "WDBX WebScraper/0.2.0"
    respect_robots_txt: true
    timeout: 10.0
    max_depth: 1
    concurrency: 5
    rate_limit: 1.0
    embedding_model: "all-MiniLM-L6-v2"

  # Ollama plugin
  ollama:
    host: "http://localhost:11434"
    model: "llama3"
    timeout: 30.0
    embedding_model: "all-MiniLM-L6-v2"

  # LMStudio plugin
  lmstudio:
    host: "localhost"
    port: 8000
    model: ""
    embedding_model: ""
    timeout: 30.0

  # Social Media plugin
  socialmedia:
    enabled_platforms: "twitter,reddit"
    cache_ttl: 300
    demo_mode: true

# Security settings
security:
  enable_encryption: false
  enable_authentication: false
  enable_access_control: false
  token_expiry: 86400 # 24 hours

# Distributed settings
distributed:
  host: "localhost"
  port: 7777
  auth_enabled: false
  auth_key: ""
  replication_factor: 1
  coordinator_host: "localhost"
  coordinator_port: 7777
```

## Quick Start

### Basic Usage

```python
from wdbx import WDBX

# Create a WDBX instance
wdbx = WDBX(
    vector_dimension=384,  # Common dimension for modern embedding models
    num_shards=2,
    data_dir="./wdbx_data",
    enable_plugins=True,
)

# Initialize the instance
import asyncio
asyncio.run(wdbx.initialize())

# Store a vector
vector = [0.1 for _ in range(384)]  # Create a 384-dimensional vector with each element set to 0.1
metadata = {"source": "example", "content": "Sample text"}
vector_id = wdbx.vector_store(vector, metadata)

# Search for similar vectors
results = wdbx.vector_search(vector, limit=5)
for vector_id, similarity, metadata in results:
    print(f"Vector ID: {vector_id}, Similarity: {similarity:.4f}")
    print(f"Content: {metadata.get('content')}")

# Don't forget to close the database
asyncio.run(wdbx.shutdown())
```

### Asynchronous API

```python
import asyncio
from wdbx import WDBX

async def main():
    # Create and initialize WDBX instance
    wdbx = WDBX(vector_dimension=384)
    await wdbx.initialize()

    # Store vectors asynchronously
    vector_id = await wdbx.vector_store_async([0.1 for _ in range(384)], {"text": "Example"})

    # Search asynchronously
    results = await wdbx.vector_search_async([0.1 for _ in range(384)], limit=5)

    # Clean up
    await wdbx.shutdown()

# Run the async function
asyncio.run(main())
```

### Using Plugins

```python
from wdbx import WDBX

# Create WDBX with plugins enabled
wdbx = WDBX(vector_dimension=384, enable_plugins=True)

# Initialize the instance
import asyncio
asyncio.run(wdbx.initialize())

# Get a plugin instance
webscraper = wdbx.get_plugin("webscraper")

# Use the plugin to extract content and create an embedding
content = asyncio.run(webscraper.extract_content("https://example.com"))
embedding = asyncio.run(webscraper.create_embedding(content))

# Store in the database
metadata = {"url": "https://example.com", "content": content}
vector_id = wdbx.vector_store(embedding, metadata)

# Clean up
asyncio.run(wdbx.shutdown())
```

### Using the CLI

The Command-Line Interface provides easy access to WDBX functionality:

```bash
# Display help
wdbx help

# Store a vector from text
wdbx store --from-text "This is a sample text to embed"

# Search for similar vectors
wdbx search --from-text "sample text" --limit 5

# Start the API server
wdbx serve --port 8000
```

### Starting the API Server

```python
from wdbx import WDBX
from wdbx.api import WDBXAPIServer
import asyncio

async def main():
    # Create and initialize WDBX
    wdbx = WDBX(vector_dimension=384, enable_plugins=True)
    await wdbx.initialize()

    # Create and start API server
    server = WDBXAPIServer(wdbx, port=8000)
    await server.initialize()
    await server.start()

# Run the server
asyncio.run(main())
```

## Components

### Core System

- **Vector Storage**: High-performance storage for vector embeddings
- **Indexing**: Multiple indexing options (HNSW, Faiss) for efficient similarity search
- **Distributed Architecture**: Sharding and replication for scalability and fault tolerance
- **Configuration Management**: Flexible configuration system with environment variables and config files

### Plugins

WDBX includes several plugins for integration with external services:

| Plugin | Description | Status |
|--------|-------------|--------|
| WebScraper | Web content extraction and analysis | Stable |
| Ollama | Local LLM integration via Ollama API | Stable |
| LMStudio | OpenAI-compatible local API integration | Stable |
| Discord | Chat integration with Discord | Stable |
| Twitch | Twitch chat and API integration | Stable |
| YouTube | YouTube data and analytics | Stable |
| SocialMedia | Cross-platform social media integration | Stable |

### Utilities

- **Visualization**: Tools for visualizing vector spaces and relationships
- **Security**: Authentication, encryption, and access control features
- **API Server**: RESTful API for remote access to WDBX functionality
- **CLI**: Command-line interface for easy management

## API Endpoints

The WDBX API server provides the following endpoints:

### Health Check

- **GET /api/v1/health**: Check the health of the API server.

### Vector Operations

- **POST /api/v1/vectors**: Store a vector.
- **POST /api/v1/vectors/search**: Search for similar vectors.
- **GET /api/v1/vectors/{vector_id}**: Get a vector by ID.
- **DELETE /api/v1/vectors/{vector_id}**: Delete a vector.
- **PUT /api/v1/vectors/{vector_id}/metadata**: Update vector metadata.

### Database Operations

- **GET /api/v1/stats**: Get database statistics.
- **POST /api/v1/clear**: Clear the database.

### Embedding Operations

- **POST /api/v1/embeddings**: Create an embedding for a text.
- **POST /api/v1/embeddings/batch**: Create embeddings for a batch of texts.

### Plugin Operations

- **GET /api/v1/plugins**: List available plugins.
- **GET /api/v1/plugins/{plugin_name}**: Get information about a plugin.

## Documentation

Comprehensive documentation is available in the [docs](docs/) directory:

- **API Reference**: Detailed class and method references
- **Plugin System**: How the plugin system works
- **Security Guide**: Authentication and encryption features
- **Visualization Guide**: Tools for visualizing vector data
- **CLI Reference**: Command-line interface documentation

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/donaldfilimon/wdbx-py.git
cd wdbx-py

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt -U

# Set up pre-commit hooks
pre-commit install
```

## Testing

Run the test suite:

```bash
# Run core tests
pytest

# Run plugin-specific tests
python wdbx/tests.test_core.py -v
python wdbx/tests.test_plugins.py -v
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

WDBX is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
