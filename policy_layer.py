#!/usr/bin/env python3
"""
Memory Policy Layer for OpenClaw

This module enforces write and retrieval policies on the Raspberry Pi side.
No infrastructure changes - purely logic layer.

Policies defined in:
- MEMORY_WRITE_POLICY.md
- MEMORY_RETRIEVAL_POLICY.md
"""

import hashlib
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# POLICY CONSTANTS
# ============================================================================

# Write Policy Constants
MIN_IMPORTANCE = 0.7
ALLOWED_KINDS = {"fact", "preference", "task"}
VALID_SCOPES = {"global", "local", "session"}
DEFAULT_SCOPE = "global"
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 2000

# Retrieval Policy Constants
DEFAULT_LIMIT = 5
MAX_LIMIT = 20
SIMILARITY_THRESHOLD = 0.30
SCOPE_BOOSTS = {
    "global": 0.05,
    "local": 0.0,
    "session": -0.02
}
ALLOWED_RESULT_FIELDS = {"text", "score", "kind", "scope", "importance", "timestamp"}


# ============================================================================
# WRITE POLICY FUNCTIONS
# ============================================================================

def should_persist(item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Determine if a memory item should be persisted based on write policy.
    
    Args:
        item: Memory item dict with keys like text, kind, scope, importance
        
    Returns:
        Tuple of (should_persist: bool, reason: Optional[str])
        - If should_persist is True, reason is None
        - If should_persist is False, reason explains why
    """
    # Check 1: Importance threshold
    importance = item.get("importance", 0.5)
    if importance < MIN_IMPORTANCE:
        return False, f"Importance {importance} below threshold {MIN_IMPORTANCE}"
    
    # Check 2: Valid kind
    kind = item.get("kind", "")
    if kind not in ALLOWED_KINDS:
        return False, f"Invalid kind '{kind}'. Allowed: {ALLOWED_KINDS}"
    
    # Check 3: Valid scope (coerce if needed, don't reject)
    scope = item.get("scope", DEFAULT_SCOPE)
    if scope not in VALID_SCOPES:
        logger.warning(f"Invalid scope '{scope}', coercing to '{DEFAULT_SCOPE}'")
        item["scope"] = DEFAULT_SCOPE
    
    # Check 4: Text content quality
    text = item.get("text", "")
    if len(text) < MIN_TEXT_LENGTH:
        return False, f"Text too short ({len(text)} chars, min {MIN_TEXT_LENGTH})"
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})"
    
    # Check 5: Not raw tool output or log (heuristic check)
    prohibited_patterns = [
        r'^\[?\d{4}-\d{2}-\d{2}.*\]?',  # Log timestamps
        r'^(DEBUG|INFO|WARN|ERROR|TRACE)',  # Log levels
        r'^\$\s+',  # Shell prompts
        r'^>',  # Tool output markers
        r'^(stdout|stderr):',  # Raw output labels
    ]
    
    for pattern in prohibited_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False, f"Text appears to be raw output/log (matches pattern: {pattern})"
    
    # All checks passed
    return True, None


def generate_memory_id(item: Dict[str, Any]) -> str:
    """
    Generate deterministic memory ID from content hash.
    
    Args:
        item: Memory item
        
    Returns:
        16-character hex hash
    """
    # Create deterministic string from key fields
    content = f"{item.get('text', '')}:{item.get('scope', '')}:{item.get('kind', '')}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def check_duplicate(existing_ids: set, item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if memory item is a duplicate.
    
    Args:
        existing_ids: Set of existing memory IDs
        item: Memory item to check
        
    Returns:
        Tuple of (is_duplicate: bool, memory_id: Optional[str])
    """
    memory_id = generate_memory_id(item)
    return memory_id in existing_ids, memory_id


# ============================================================================
# RETRIEVAL POLICY FUNCTIONS
# ============================================================================

def score_memory(result: Dict[str, Any]) -> float:
    """
    Calculate composite score for a memory result.
    
    Formula: (similarity_score × importance) + scope_boost
    
    Args:
        result: Memory result from search
        
    Returns:
        Composite score
    """
    similarity = result.get("score", 0.0)
    importance = result.get("importance", 0.5)
    scope = result.get("scope", "local")
    
    scope_boost = SCOPE_BOOSTS.get(scope, 0.0)
    
    # Apply temporal decay to old session memories
    temporal_decay = 1.0
    if scope == "session":
        timestamp = result.get("timestamp")
        if timestamp:
            try:
                # Parse timestamp and calculate age
                if isinstance(timestamp, str):
                    # Handle ISO format
                    ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    ts = timestamp
                
                age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
                if age_hours > 24:
                    # Decay over a week (168 hours)
                    temporal_decay = max(0.9, 1.0 - (age_hours - 24) / 168)
            except (ValueError, TypeError):
                pass  # If timestamp parsing fails, no decay
    
    composite = (similarity * importance * temporal_decay) + scope_boost
    return max(0.0, min(1.0, composite))  # Clamp to 0-1


def normalize_text(text: str) -> str:
    """
    Normalize text for deduplication.
    
    Args:
        text: Raw text
        
    Returns:
        Normalized text
    """
    # Lowercase, strip whitespace, remove punctuation
    normalized = text.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def compute_content_hash(text: str) -> str:
    """
    Compute hash of normalized content for deduplication.
    
    Args:
        text: Text to hash
        
    Returns:
        16-character hex hash
    """
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def deduplicate_memories(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate memories by content hash, keeping highest scoring.
    
    Args:
        results: List of memory results
        
    Returns:
        Deduplicated list
    """
    seen_hashes = {}
    
    # Sort by original score (descending) to keep best first
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    for result in sorted_results:
        text = result.get("text", "")
        content_hash = compute_content_hash(text)
        
        if content_hash not in seen_hashes:
            seen_hashes[content_hash] = result
    
    return list(seen_hashes.values())


def filter_retrieval(results: List[Dict[str, Any]], 
                     limit: int = DEFAULT_LIMIT,
                     min_score: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Apply full retrieval policy to search results.
    
    Steps:
    1. Filter by similarity threshold
    2. Deduplicate
    3. Calculate composite scores
    4. Sort by composite score
    5. Limit results
    6. Filter payload fields
    
    Args:
        results: Raw results from memory search
        limit: Maximum results to return (clamped to 1-MAX_LIMIT)
        min_score: Minimum similarity score
        
    Returns:
        Filtered, scored, and limited results
    """
    # Clamp limit
    limit = max(1, min(limit, MAX_LIMIT))
    
    # Step 1: Filter by similarity threshold
    filtered = [r for r in results if r.get("score", 0) >= min_score]
    
    if not filtered:
        return []
    
    # Step 2: Deduplicate
    deduplicated = deduplicate_memories(filtered)
    
    # Step 3: Calculate composite scores
    for result in deduplicated:
        result["composite_score"] = score_memory(result)
    
    # Step 4: Sort by composite score (descending)
    sorted_results = sorted(deduplicated, key=lambda x: x.get("composite_score", 0), reverse=True)
    
    # Step 5: Limit results
    limited = sorted_results[:limit]
    
    # Step 6: Filter payload fields and use composite_score as final score
    cleaned = []
    for result in limited:
        cleaned_result = {k: v for k, v in result.items() if k in ALLOWED_RESULT_FIELDS}
        cleaned_result["score"] = result.get("composite_score", result.get("score", 0))
        cleaned.append(cleaned_result)
    
    return cleaned


# ============================================================================
# POLICY STATISTICS
# ============================================================================

class PolicyStats:
    """Track policy enforcement statistics"""
    
    def __init__(self):
        self.writes_attempted = 0
        self.writes_accepted = 0
        self.writes_rejected = {}
        self.retrievals_processed = 0
        self.duplicates_filtered = 0
    
    def record_write_attempt(self, accepted: bool, reason: Optional[str] = None):
        """Record a write attempt"""
        self.writes_attempted += 1
        if accepted:
            self.writes_accepted += 1
        else:
            self.writes_rejected[reason] = self.writes_rejected.get(reason, 0) + 1
    
    def record_retrieval(self, raw_count: int, filtered_count: int):
        """Record a retrieval operation"""
        self.retrievals_processed += 1
        self.duplicates_filtered += (raw_count - filtered_count)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "writes_attempted": self.writes_attempted,
            "writes_accepted": self.writes_accepted,
            "writes_rejected": self.writes_rejected,
            "write_acceptance_rate": self.writes_accepted / max(1, self.writes_attempted),
            "retrievals_processed": self.retrievals_processed,
            "duplicates_filtered": self.duplicates_filtered
        }


# Global stats instance
_stats = PolicyStats()


def get_policy_stats() -> Dict[str, Any]:
    """Get policy enforcement statistics"""
    return _stats.get_stats()


def reset_policy_stats():
    """Reset policy statistics"""
    global _stats
    _stats.writes_attempted = 0
    _stats.writes_accepted = 0
    _stats.writes_rejected = {}
    _stats.retrievals_processed = 0
    _stats.duplicates_filtered = 0


# ============================================================================
# CONVENIENCE FUNCTIONS FOR WRAPPER
# ============================================================================

def apply_write_policy(item: Dict[str, Any], 
                       existing_ids: Optional[set] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Apply full write policy and return decision.
    
    Args:
        item: Memory item
        existing_ids: Optional set of existing memory IDs for duplicate check
        
    Returns:
        Tuple of (should_persist, reason, memory_id)
    """
    # Check basic policy
    should_persist, reason = should_persist_check(item)
    
    if not should_persist:
        _stats.record_write_attempt(False, reason)
        return False, reason, None
    
    # Check duplicates if existing_ids provided
    memory_id = generate_memory_id(item)
    if existing_ids and memory_id in existing_ids:
        _stats.record_write_attempt(False, "duplicate")
        return False, "Duplicate memory ID", memory_id
    
    _stats.record_write_attempt(True)
    return True, None, memory_id


def should_persist_check(item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Alias for should_persist for clarity"""
    return should_persist(item)


if __name__ == "__main__":
    # Quick self-test
    print("Policy Layer Self-Test\n")
    
    # Test write policy
    print("Write Policy Tests:")
    
    test_cases = [
        {"text": "User likes coffee", "kind": "preference", "scope": "global", "importance": 0.8},
        {"text": "Short", "kind": "fact", "scope": "global", "importance": 0.9},  # Too short
        {"text": "Low importance item", "kind": "fact", "scope": "global", "importance": 0.5},  # Too low
        {"text": "DEBUG: Some log message", "kind": "fact", "scope": "global", "importance": 0.9},  # Raw log
    ]
    
    for i, item in enumerate(test_cases):
        should, reason = should_persist(item)
        status = "ACCEPT" if should else "REJECT"
        print(f"  {i+1}. {status}: {item['text'][:30]}... ({reason or 'OK'})")
    
    print("\nRetrieval Policy Tests:")
    
    # Test retrieval policy
    test_results = [
        {"text": "User likes coffee", "score": 0.85, "importance": 0.9, "scope": "global", "kind": "preference"},
        {"text": "User likes coffee", "score": 0.82, "importance": 0.8, "scope": "global", "kind": "preference"},  # Dup
        {"text": "Some old info", "score": 0.60, "importance": 0.9, "scope": "global", "kind": "fact"},  # Below threshold
        {"text": "Project context", "score": 0.80, "importance": 0.7, "scope": "local", "kind": "fact"},
    ]
    
    filtered = filter_retrieval(test_results, limit=3)
    print(f"  Input: {len(test_results)} results")
    print(f"  Output: {len(filtered)} results")
    for r in filtered:
        print(f"    - {r['text'][:30]}... (score: {r['score']:.3f})")
    
    print("\nPolicy Stats:", get_policy_stats())
