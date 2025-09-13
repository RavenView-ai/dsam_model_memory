# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for llama-cpp-python and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt ./

# Install Python dependencies with proper llama-cpp-python compilation
# Default to CUDA support for GPU acceleration (falls back to CPU if CUDA not available)
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AM_DB_PATH=/app/data/memories.db
ENV AM_INDEX_PATH=/app/data/faiss_index
ENV AM_LOG_DIR=/app/logs

# Expose ports for all services
EXPOSE 5001 8000 8001

# Default command to start all servers
CMD ["python", "-m", "agentic_memory.cli", "server", "start", "--all"]
