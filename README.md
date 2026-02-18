## Ghost Memory

AI Memory System with vector storage and semantic retrieval. Built for personal AI assistants to maintain long-term memory across conversations.

**🌐 Live Demo:** https://ghostenvoy.github.io/ghost-memory/

![Ghost Memory](https://ghostenvoy.github.io/ghost-memory/)

## Features

- **Qdrant** vector database for semantic memory storage
- **Embedding service** for converting text to vectors  
- **Memory middleware** for retrieval/write with importance scoring
- **Policy layer** for controlling what gets stored
- **Mode router** for different memory retrieval strategies

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/GhostEnvoy/ghost-memory.git
cd ghost-memory
```

### 2. Start Infrastructure

```bash
cd memory-node
docker-compose up -d
```

This starts:
- Qdrant on port 6333
- Embedding service on port 8040

### 3. Configure Environment

```bash
export MEMORY_NODE_URL=http://localhost:6333
export EMBEDDING_URL=http://localhost:8040
```

### 4. Use the Memory System

```python
from memory_middleware import memory_middleware

# Store a memory
memory_middleware(
    action="write",
    text="Learned about neural networks today",
    importance=0.8,
    kind="fact"
)

# Retrieve memories
results = memory_middleware(
    action="retrieve",
    query="what did we discuss about AI?"
)
```

## Architecture

```
ghost-memory/
├── memory_middleware.py      # Main API entry point
├── memory_client.py          # Qdrant client wrapper
├── memory_controller.py      # Memory orchestration
├── memory_wrapper.py         # High-level wrapper
├── memory_controller_config.py
├── policy_layer.py          # Storage policies
├── policy_layer_enforcement.py
├── mode_router.py           # Retrieval strategies
├── architecture_introspection.py
└── memory-node/
    ├── docker-compose.yml   # Infrastructure setup
    └── README.md
```

## Components

| File | Purpose |
|------|---------|
| `memory_middleware.py` | Main API for storing and retrieving memories |
| `memory_client.py` | Direct Qdrant database operations |
| `memory_controller.py` | Orchestrates retrieval/write flow |
| `policy_layer.py` | Controls what gets stored (importance thresholds) |
| `mode_router.py` | Different retrieval modes (semantic, keyword, etc.) |

## Configuration

Environment variables:
- `MEMORY_NODE_URL` - Qdrant HTTP URL (default: http://localhost:6333)
- `EMBEDDING_URL` - Embedding service URL (default: http://localhost:8040)

## Use Cases

- Personal AI assistant memory
- Chatbot conversation history
- Knowledge base retrieval
- Context-aware AI agents

## License

MIT - Created by Julius as a portfolio project.
