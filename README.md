# JAM (Journalistic Agent Memory)

A local-first memory system for LLM agents that provides persistent, searchable memory using journalistic 5W1H (Who, What, When, Where, Why, How) semantic extraction. Every conversation, tool use, and interaction becomes a searchable memory that the AI can recall and build upon.

## Overview

JAM transforms AI interactions into structured, queryable memories using journalistic principles. The system runs entirely locally using llama.cpp and provides both web and API interfaces for integration.

### Key Features

- **5W1H Memory Extraction**: Automatically decomposes events into journalistic dimensions
- **Dual-Model Architecture**: Separate LLM and embedding models for optimized performance
- **Hybrid Retrieval**: Six-strategy retrieval system with configurable weights
- **Token-Optimized Selection**: Greedy knapsack algorithm for maximizing context utility
- **Local-First Design**: Complete privacy with all processing on your machine
- **Web Interface**: Browser UI for memory management, chat, and visualization
- **OpenAI-Compatible API**: Drop-in replacement for external API clients
- **3D Memory Visualization**: Interactive topology exploration using PCA/t-SNE/UMAP
- **Document Processing**: Support for PDF, DOCX, HTML, CSV, and more formats

## Quick Start

### Prerequisites

- Python 3.11+
- 16GB+ RAM recommended
- GGUF model files (main LLM and embedding model)
- Windows, macOS, or Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jam_model_memory.git
cd jam_model_memory

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install document parsing support
pip install -r requirements-optional.txt
```

#### GPU Acceleration (Optional)

```bash
# For CUDA support
CMAKE_ARGS="-DGGML_CUDA=on"

# For Metal support (macOS)
CMAKE_ARGS="-DLLAMA_METAL=on"
```

### Configuration

1. Copy the example configuration:
```bash
cp .env.example .env
```

2. Edit `.env` to set your model paths:
```env
# Main LLM model
AM_MODEL_PATH=/path/to/your/llm_model.gguf

# Embedding model
AM_EMBEDDING_MODEL_PATH=/path/to/embedding_model.gguf
AM_EMBEDDING_DIMENSIONS=1024  # Match your embedding model's dimensions
```

### Running the Application

Start all services:

```bash
python -m agentic_memory.cli server start --all
```

Services will be available at:
- **Web Interface**: http://localhost:5001
- **API Server**: http://localhost:8001
- **LLM Server**: http://localhost:8000
- **Embedding Server**: http://localhost:8002

## Architecture

### Core Components

#### Memory System
- **MemoryRouter**: Central orchestrator for ingestion, retrieval, and chat operations
- **HybridRetriever**: Six-strategy retrieval combining semantic, recency, actor, temporal, spatial, and usage signals
- **BlockBuilder**: Token-budgeted memory selection using greedy knapsack algorithm
- **5W1H Extractors**: LLM-based extraction pipeline for journalistic dimensions

#### Storage Layer
- **MemoryStore**: SQLite database with FTS5 full-text search
- **FaissIndex**: High-performance vector similarity search
- **ConfigManager**: Runtime configuration persistence

#### Server Infrastructure
- **LLMServerManager**: Auto-manages main llama.cpp server lifecycle
- **EmbeddingServerManager**: Dedicated embedding generation service
- **ProcessManager**: Background process management with PID tracking
- **Flask Application**: Web interface and REST API endpoints

### Dual-Model System

JAM uses two separate models for optimal performance:

1. **Main LLM**: Handles reasoning, extraction, and chat responses
2. **Embedding Model**: Generates vector representations for semantic search

This separation allows using a smaller, faster model for embeddings while maintaining a powerful LLM for complex tasks.

## Usage

### Web Interface

Navigate to http://localhost:5001 to access:

- **Analyzer** (`/`): Advanced memory search with token budget management
- **Chat** (`/chat`): Interactive AI assistant with automatic memory augmentation
- **Browser** (`/browser`): Browse and filter all stored memories
- **Analytics** (`/analytics`): Memory statistics and usage patterns
- **3D Visualization** (`/visualize`): Interactive memory topology exploration
- **Configuration** (`/config`): Runtime settings and weight adjustment

### Chat Interface Features

- **Automatic Memory Retrieval**: Relevant memories automatically augment responses
- **Context Transparency**: View which memories inform each response
- **Manual Memory Selection**: Search and select specific memories for context
- **Conversation Persistence**: All interactions saved as searchable memories
- **Token Budget Display**: Real-time tracking of context window usage

### CLI Commands

#### Memory Operations
```bash
# Add a memory
python -m agentic_memory.cli memory add "Meeting notes from project discussion"

# Search memories
python -m agentic_memory.cli memory search "project discussion" --limit 10

# View statistics
python -m agentic_memory.cli memory stats

# Bulk import memories
python -m agentic_memory.cli memory import memories.jsonl
```

#### Server Management
```bash
# Start services
python -m agentic_memory.cli server start --all      # All services
python -m agentic_memory.cli server start --web      # Web interface only
python -m agentic_memory.cli server start --api      # API server only
python -m agentic_memory.cli server start --llm      # LLM server only
python -m agentic_memory.cli server start --embedding # Embedding server only

# Check status
python -m agentic_memory.cli server status

# Stop services
python -m agentic_memory.cli server stop --all
```

#### Document Processing
```bash
# Parse single document
python -m agentic_memory.cli document parse /path/to/document.pdf

# Batch process directory
python -m agentic_memory.cli document batch /path/to/documents --recursive

# List supported formats
python -m agentic_memory.cli document formats
```

### Python API

```python
from agentic_memory import MemoryRouter

# Initialize memory system
router = MemoryRouter()

# Add memories
await router.ingest("Important meeting with team about Q4 planning")

# Search memories
results = await router.retrieve("team meetings", k=10)

# Chat with memory context
response = await router.chat("What were our Q4 planning decisions?")

# Get statistics
stats = await router.get_memory_stats()
```

### API Endpoints

The OpenAI-compatible API provides:

- `POST /v1/chat/completions` - Chat with memory augmentation
- `POST /v1/completions` - Text completion with memory context
- `GET /v1/models` - List available models
- `POST /v1/embeddings` - Generate embeddings

## Configuration

### Essential Settings

```env
# Model Configuration
AM_MODEL_PATH=/path/to/llm_model.gguf
AM_EMBEDDING_MODEL_PATH=/path/to/embedding_model.gguf
AM_EMBEDDING_DIMENSIONS=1024  # 384, 768, 1024, or 1536

# Performance Settings
AM_CONTEXT_WINDOW=8192
AM_GPU_LAYERS=-1  # -1 for all layers, 0 for CPU only
AM_THREADS=8
AM_BATCH_SIZE=8192

# Storage Paths
AM_DB_PATH=./data/amemory.sqlite3
AM_INDEX_PATH=./data/faiss.index
```

### Advanced Configuration

JAM supports any llama.cpp server flag through environment variables:

```env
# Format: AM_LLAMA_FLAG_<flag_name> for LLM server
#         AM_EMBEDDING_FLAG_<flag_name> for embedding server

# Performance Optimization
AM_LLAMA_FLAG_flash_attn=true  # Flash attention for supported GPUs
AM_LLAMA_FLAG_mlock=true       # Lock model in memory
AM_LLAMA_FLAG_no_mmap=true     # Keep model in VRAM

# Cache Optimization
AM_LLAMA_FLAG_cache_type_k=q8_0  # Quantized KV cache
AM_LLAMA_FLAG_cache_type_v=q4_0  # Further V cache quantization

# Context Extension
AM_LLAMA_FLAG_rope_scaling=yarn  # YaRN for better long context
AM_LLAMA_FLAG_rope_freq_base=10000

# Monitoring
AM_LLAMA_FLAG_metrics=true  # Prometheus metrics endpoint
AM_LLAMA_FLAG_verbose=true  # Detailed logging
```

### Retrieval Weights

Configure the hybrid retrieval strategy (must sum to 1.0):

```env
AM_W_SEMANTIC=0.84   # Semantic similarity
AM_W_RECENCY=0.00    # Time-based relevance
AM_W_ACTOR=0.00      # Actor relevance
AM_W_TEMPORAL=0.08   # Temporal similarity
AM_W_SPATIAL=0.04    # Location relevance
AM_W_USAGE=0.04      # Access frequency
```

## Features

### Token-Optimized Memory Selection

The greedy knapsack algorithm optimizes memory selection within token constraints:

1. **Utility Scoring**: Each memory has a retrieval score (relevance)
2. **Token Cost**: Each memory's token count is calculated
3. **Optimization**: Maximizes total utility within token budget
4. **Efficiency**: Greedy approach provides near-optimal results quickly

### 5W1H Extraction Pipeline

Memories are extracted with journalistic precision:

- **Who**: Actors and entities involved
- **What**: Core events and actions
- **When**: Temporal information and timestamps
- **Where**: Locations and spatial context
- **Why**: Motivations and reasons
- **How**: Methods and processes

### Embedding Generation

The dedicated embedding server provides:

- Fast vector generation for semantic search
- Support for multiple embedding dimensions
- Configurable pooling strategies (mean, CLS)
- Batch processing for efficiency
- Caching for frequently accessed embeddings

## Project Structure

```
jam_model_memory/
├── agentic_memory/
│   ├── extraction/        # 5W1H extraction pipeline
│   ├── storage/           # Database and vector stores
│   ├── server/            # Web and API servers
│   ├── embedding/         # Embedding generation
│   ├── tools/             # Tool integrations
│   ├── cli.py             # Command-line interface
│   ├── router.py          # Main orchestrator
│   ├── retrieval.py       # Hybrid retrieval system
│   ├── block_builder.py   # Token optimization
│   └── config_manager.py  # Configuration management
├── data/                  # Databases and indexes
├── logs/                  # Application logs
└── requirements.txt       # Python dependencies
```

## Database Schema

SQLite database with optimized tables:

- **memories**: Core storage with 5W1H fields, embeddings, and metadata
- **memory_actors**: Many-to-many actor associations
- **memory_locations**: Many-to-many location associations
- **memory_scores**: Cached retrieval scores for performance
- **config**: Persisted runtime configuration
- **FTS5 tables**: Full-text search indexes

## Performance Optimization

### Hardware Profiles

Choose settings based on your hardware:

- **High-End** (RTX 4090/3090, 32GB+ RAM): Full GPU acceleration, large context
- **Mid-Range** (RTX 3070/3060, 16GB RAM): Partial GPU layers, medium context
- **Low-End** (GTX 1660/CPU, 8GB RAM): CPU inference, smaller context
- **Apple Silicon**: Metal acceleration, unified memory advantages

### Optimization Tips

1. **Model Selection**: Use quantized models (Q4_K_M, Q5_K_S) for better speed/quality balance
2. **Embedding Model**: Use smaller embedding models (0.3B-0.6B parameters)
3. **Context Window**: Adjust based on available memory
4. **Batch Size**: Increase for throughput, decrease for latency
5. **Cache Quantization**: Enable for memory savings with minimal quality loss

## Troubleshooting

### Common Issues

**Server won't start**
- Check ports 5001, 8000, 8001, 8002 are available
- Verify model paths in .env are correct
- Ensure sufficient system memory

**Slow performance**
- Enable GPU acceleration if available
- Reduce context window size
- Use smaller or more quantized models
- Enable flash attention for supported GPUs

**Memory errors**
- Reduce batch size and context window
- Enable cache quantization
- Use system memory monitoring tools

**Embedding dimension mismatch**
- Ensure AM_EMBEDDING_DIMENSIONS matches your embedding model
- Rebuild FAISS index if changing embedding models

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient local inference
- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- Inspired by journalistic principles of information organization
