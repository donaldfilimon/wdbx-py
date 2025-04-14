FROM python:3.11-slim

LABEL maintainer="WDBX Team <info@wdbx.ai>"
LABEL description="WDBX Vector Database for AI Applications"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements*.txt ./

# Install core requirements and minimal plugins
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/data /app/config

# Copy application
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WDBX_DATA_DIR=/app/data
ENV WDBX_LOG_LEVEL=INFO

# Expose API port
EXPOSE 8000

# Set default command
CMD ["python", "-m", "wdbx.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

# Usage:
# Build: docker build -t wdbx:latest .
# Run API server: docker run -p 8000:8000 -v $(pwd)/data:/app/data wdbx:latest
# Run CLI: docker run -it --rm -v $(pwd)/data:/app/data wdbx:latest python -m wdbx.cli
