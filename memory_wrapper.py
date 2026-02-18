#!/usr/bin/env python3
"""
OpenClaw Memory Wrapper
Simple interface for OpenClaw to interact with external memory node

Usage:
    from memory_wrapper import remember, recall, check_memory
    
    # Store a memory
    remember("User prefers dark mode", scope="user_prefs", importance=0.8)
    
    # Recall memories
    results = recall("dark mode preference")
    
    # Check if memory is working
    if check_memory():
        print("Memory node is healthy")
"""

import os
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

# Add workspace to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_client import (
    memory_health,
    memory_upsert,
    memory_search,
    MemoryClientError
)

from policy_layer import (
    should_persist,
    filter_retrieval,
    get_policy_stats,
    SIMILARITY_THRESHOLD,
    DEFAULT_LIMIT,
    _stats
)

# Default values
DEFAULT_SCOPE = "global"
DEFAULT_KIND = "fact"
DEFAULT_IMPORTANCE = 0.5

# Valid scopes as enforced by memory node
VALID_SCOPES = ["global", "local", "session"]


def check_memory() -> bool:
    """
    Quick health check for memory node
    
    Returns:
        True if memory node is accessible and healthy
    """
    try:
        health = memory_health()
        return health.get('status') == 'healthy'
    except Exception:
        return False


def remember(
    text: str,
    scope: str = DEFAULT_SCOPE,
    kind: str = DEFAULT_KIND,
    importance: float = DEFAULT_IMPORTANCE,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Store a memory item (with policy enforcement)
    
    Args:
        text: The content to remember
        scope: Memory scope ('global', 'local', or 'session')
        kind: Type of memory (e.g., 'fact', 'preference', 'task')
        importance: Importance score 0.0-1.0
        metadata: Optional additional metadata
        
    Returns:
        True if successfully stored
    """
    try:
        # Validate scope - coerce to valid value if needed
        if scope not in VALID_SCOPES:
            scope = DEFAULT_SCOPE
        
        item = {
            "text": text,
            "scope": scope,
            "kind": kind,
            "importance": importance,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if metadata:
            item.update(metadata)
        
        # Apply write policy
        should, reason = should_persist(item)
        if not should:
            print(f"[Memory] Policy rejected: {reason}")
            _stats.record_write_attempt(False, reason)
            return False
        
        result = memory_upsert([item])
        success = result.get('inserted', 0) >= 1
        if success:
            _stats.record_write_attempt(True)
        return success
        
    except MemoryClientError as e:
        print(f"[Memory] Failed to store: {e}")
        return False
    except Exception as e:
        print(f"[Memory] Unexpected error: {e}")
        return False


def recall(
    query: str,
    limit: int = DEFAULT_LIMIT,
    min_score: float = SIMILARITY_THRESHOLD,
    apply_policy: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for relevant memories (with policy enforcement)
    
    Args:
        query: Search query
        limit: Maximum number of results (default: 5, max: 20)
        min_score: Minimum similarity score (default: 0.75)
        apply_policy: Whether to apply retrieval policy filtering
        
    Returns:
        List of memory items with scores
    """
    try:
        # Request more results to allow for policy filtering
        search_limit = limit * 3 if apply_policy else limit
        result = memory_search(query, limit=search_limit)
        results = result.get('results', [])
        
        raw_count = len(results)
        
        if apply_policy and results:
            # Apply full retrieval policy
            results = filter_retrieval(results, limit=limit, min_score=min_score)
        elif min_score > 0:
            # Simple score filtering if policy not applied
            results = [r for r in results if r.get('score', 0) >= min_score][:limit]
        else:
            results = results[:limit]
        
        # Track retrieval stats
        _stats.record_retrieval(raw_count, len(results))
        
        return results
        
    except MemoryClientError as e:
        print(f"[Memory] Failed to search: {e}")
        return []
    except Exception as e:
        print(f"[Memory] Unexpected error: {e}")
        return []


def get_memory_stats(include_policy: bool = True) -> Dict[str, Any]:
    """
    Get memory node statistics
    
    Args:
        include_policy: Whether to include policy enforcement stats
        
    Returns:
        Dict with health info, collection stats, and optionally policy stats
    """
    try:
        health = memory_health()
        stats = {
            "healthy": health.get('status') == 'healthy',
            "collection": health.get('collection', {}),
            "version": health.get('version'),
            "qdrant_version": health.get('qdrant_version')
        }
        
        if include_policy:
            stats["policy"] = get_policy_stats()
        
        return stats
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Quick demo/test
    print("Memory Wrapper Test\n")
    
    # Check health
    if check_memory():
        print("✓ Memory node is healthy")
    else:
        print("✗ Memory node is not accessible")
        sys.exit(1)
    
    # Remember something
    if remember("Test memory from OpenClaw wrapper", scope="global", importance=0.5):
        print("✓ Memory stored successfully")
    else:
        print("✗ Failed to store memory")
    
    # Recall
    results = recall("OpenClaw wrapper test", limit=3)
    print(f"✓ Found {len(results)} relevant memories")
    for r in results:
        print(f"  - {r.get('text', 'N/A')[:50]}... (score: {r.get('score', 0):.3f})")
    
    # Stats
    stats = get_memory_stats()
    print(f"\nMemory Stats:")
    print(f"  Points: {stats.get('collection', {}).get('points_count')}")
    print(f"  Dimension: {stats.get('collection', {}).get('vector_size')}")
