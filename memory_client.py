#!/usr/bin/env python3
"""
Memory Client Module for OpenClaw Raspberry Pi
Connects to external Memory Node via Tailscale
"""

import os
import requests
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MEMORY_NODE_URL = os.environ.get("MEMORY_NODE_URL", "http://100.103.241.63:8000")
MEMORY_NODE_API_KEY = os.environ.get("MEMORY_NODE_API_KEY", "change-this-in-production")

# HTTP settings
TIMEOUT = 10
HEADERS = {
    "X-API-Key": MEMORY_NODE_API_KEY,
    "Content-Type": "application/json"
}


class MemoryClientError(Exception):
    """Custom exception for memory client errors"""
    pass


def _make_request(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
    """Make HTTP request to memory node with proper error handling"""
    url = f"{MEMORY_NODE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, json=data, timeout=TIMEOUT)
        else:
            raise MemoryClientError(f"Unsupported HTTP method: {method}")
        
        # Check for non-200 status codes
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error(f"Memory API error: {error_msg}")
            raise MemoryClientError(error_msg)
        
        return response.json()
        
    except requests.exceptions.Timeout:
        error_msg = f"Request to {url} timed out after {TIMEOUT}s"
        logger.error(error_msg)
        raise MemoryClientError(error_msg)
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Cannot connect to memory node at {url}: {str(e)}"
        logger.error(error_msg)
        raise MemoryClientError(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg)
        raise MemoryClientError(error_msg)


def memory_health() -> Dict:
    """
    Check memory node health status
    
    Returns:
        Dict containing health status and collection info
    """
    return _make_request("GET", "/health")


def memory_upsert(items: List[Dict]) -> Dict:
    """
    Insert or update memory items
    
    Args:
        items: List of memory items with 'text', 'scope', 'kind', 'importance' fields
        
    Returns:
        Dict with 'success' and 'count' fields
        
    Raises:
        MemoryClientError: If HTTP status != 200 or request fails
    """
    if not items:
        raise MemoryClientError("Cannot upsert empty items list")
    
    # Validate items have required fields
    required_fields = ["text"]
    for i, item in enumerate(items):
        for field in required_fields:
            if field not in item:
                raise MemoryClientError(f"Item {i} missing required field: {field}")
    
    return _make_request("POST", "/memory/upsert", {"items": items})


def memory_search(query: str, limit: int = 5) -> Dict:
    """
    Search memory for relevant items
    
    Args:
        query: Search query string
        limit: Maximum number of results (default: 5)
        
    Returns:
        Dict with 'results' containing scored memory items
        
    Raises:
        MemoryClientError: If HTTP status != 200 or request fails
    """
    if not query or not query.strip():
        raise MemoryClientError("Search query cannot be empty")
    
    if limit < 1 or limit > 100:
        raise MemoryClientError(f"Invalid limit: {limit}. Must be between 1 and 100.")
    
    payload = {
        "query": query,
        "limit": limit
    }
    
    return _make_request("POST", "/memory/search", payload)


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing memory client...")
    try:
        health = memory_health()
        print(f"Health check: {health}")
    except MemoryClientError as e:
        print(f"Health check failed: {e}")
        exit(1)
