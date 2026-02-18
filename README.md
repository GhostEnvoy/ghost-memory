# Ghost Memory

AI Memory System with vector storage and semantic retrieval.

## Features

- **Qdrant** vector database for semantic memory storage
- **Embedding service** for converting text to vectors
- **Memory middleware** for retrieval/write with importance scoring
- **Policy layer** for controlling what gets stored

## Setup

1. Install dependencies:
```bash
pip install qdrant-client sentence-transformers
```

2. Start Qdrant:
```bash
docker run -d -p 8001:6333 qdrant/qdrant:v1.9.0
```

3. Start the embedding service (see `embeddings/` folder)

4. Set environment variables:
```bash
export MEMORY_NODE_URL=http://localhost:8001
export EMBEDDING_URL=http://localhost:8040
```

## Usage

```python
from memory_middleware import memory_middleware

# Retrieve memories
results = memory_middleware(action="retrieve", query="what did we discuss about AI?")

# Store a memory
memory_middleware(action="write", text="Learned about neural networks today", importance=0.8)
```

## Architecture

- `memory_client.py` - Qdrant client wrapper
- `memory_controller.py` - Memory orchestration
- `memory_middleware.py` - Main API for retrieval/write
- `policy_layer.py` - Controls what gets stored

## License

MIT
