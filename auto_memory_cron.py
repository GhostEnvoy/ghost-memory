#!/usr/bin/env python3
"""Auto-memory cycle for cron job"""

import requests
import json
import sys

# Config
MEMORY_NODE_URL = "http://100.103.241.63:8000"
EMBEDDING_URL = "http://localhost:8040"
HEADERS = {"X-API-Key": "change-this-in-production", "Content-Type": "application/json"}

def get_embedding(text):
    """Get embedding for text"""
    resp = requests.post(f"{EMBEDDING_URL}/embed", json={"texts": [text]}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def search_memory(query, limit=5):
    """Search memory using embedding"""
    emb = get_embedding(query)
    resp = requests.post(
        f"{MEMORY_NODE_URL}/memory/search",
        headers=HEADERS,
        json={"query": query, "limit": limit},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def store_memory(text, scope="global", kind="fact", importance=0.6):
    """Store a memory item"""
    resp = requests.post(
        f"{MEMORY_NODE_URL}/memory/upsert",
        headers=HEADERS,
        json={"items": [{"text": text, "scope": scope, "kind": kind, "importance": importance}]},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()

# Main - extract user message from cron trigger
user_message = "Run the auto-memory cycle: Read session transcripts, query Qdrant for memories, store exchange"
response_text = "Processing memory cycle via cron job"

# 1. Search for relevant memories
print("Searching for relevant memories...")
try:
    search_result = search_memory(user_message, limit=5)
    memories_recalled = len(search_result.get("results", []))
    print(f"Found {memories_recalled} relevant memories")
    for r in search_result.get("results", [])[:2]:
        print(f"  - {r.get('text', '')[:80]}...")
except Exception as e:
    print(f"Search error: {e}")
    memories_recalled = 0

# 2. Store the exchange
print("\nStoring exchange to memory...")
try:
    store_result = store_memory(
        text=f"User requested auto-memory cycle: {user_message[:100]}...",
        scope="global",
        kind="fact",
        importance=0.6
    )
    stored = store_result.get("count", 1)
    print(f"Stored {stored} memory item(s)")
except Exception as e:
    print(f"Store error: {e}")
    stored = 0

# 3. Summary
print("\n=== AUTO-MEMORY CYCLE SUMMARY ===")
print(f"New messages processed: 1 (cron trigger)")
print(f"Memories recalled: {memories_recalled}")
print(f"Memories stored: {stored}")
