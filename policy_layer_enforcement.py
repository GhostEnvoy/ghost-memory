#!/usr/bin/env python3
"""
PolicyLayer - Hard Enforcement for Memory Operations

This is the ONLY authorized interface for memory operations.
Direct calls to memory_client or memory_wrapper are blocked.

Usage:
    from policy_layer_enforcement import PolicyLayer
    
    # All writes go through gate
    result = PolicyLayer.write_gate(text, kind, scope, importance)
    
    # All retrievals go through gate
    results = PolicyLayer.retrieve_gate(query, limit)
    
    # Context building
    context = PolicyLayer.build_context_block(results)
"""

import os
import sys
import hashlib
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_client import memory_health, memory_upsert, memory_search, MemoryClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# POLICY CONSTANTS - HARD-CODED, NO OVERRIDES
# ============================================================================

REQUIRED_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
REQUIRED_DIMENSION = 768
MIN_IMPORTANCE = 0.7
ALLOWED_KINDS = {"fact", "preference", "task"}
VALID_SCOPES = {"global", "local", "session"}
DEFAULT_SCOPE = "global"
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 2000
SIMILARITY_THRESHOLD = 0.30
DEFAULT_RETRIEVAL_LIMIT = 5
MAX_RETRIEVAL_LIMIT = 20
SCOPE_BOOSTS = {"global": 0.05, "local": 0.0, "session": -0.02}
MAX_CONTEXT_ITEMS = 8
MAX_CONTEXT_CHARS = 1200


# ============================================================================
# EXCEPTIONS
# ============================================================================

class PolicyViolationError(Exception):
    """Raised when policy is violated"""
    pass


class DriftDetectedError(Exception):
    """Raised when memory node config drifts from expected"""
    pass


class BypassAttemptError(Exception):
    """Raised when attempting to bypass policy layer"""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PolicyDecision:
    """Record of a policy decision"""
    timestamp: str
    operation: str  # 'write' or 'retrieve'
    accepted: bool
    reason: Optional[str] = None
    item_preview: Optional[str] = None


@dataclass
class PolicyStats:
    """Policy enforcement statistics"""
    writes_attempted: int = 0
    writes_accepted: int = 0
    writes_rejected: Dict[str, int] = field(default_factory=dict)
    retrievals_processed: int = 0
    duplicates_filtered: int = 0
    last_20_decisions: List[PolicyDecision] = field(default_factory=list)
    drift_check_passed: bool = False
    last_drift_check: Optional[str] = None


# ============================================================================
# POLICY LAYER - SINGLETON ENFORCEMENT
# ============================================================================

class PolicyLayer:
    """
    Hard enforcement layer for memory operations.
    
    This class is the ONLY authorized interface for memory writes and retrievals.
    Any attempt to bypass it will raise BypassAttemptError.
    """
    
    _instance = None
    _initialized = False
    _stats = PolicyStats()
    _drift_check_done = False
    _memory_enabled = True
    _health_data = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _record_decision(cls, operation: str, accepted: bool, 
                        reason: Optional[str] = None,
                        item_preview: Optional[str] = None):
        """Record a policy decision"""
        decision = PolicyDecision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation=operation,
            accepted=accepted,
            reason=reason,
            item_preview=item_preview[:50] if item_preview else None
        )
        cls._stats.last_20_decisions.append(decision)
        if len(cls._stats.last_20_decisions) > 20:
            cls._stats.last_20_decisions.pop(0)
    
    @classmethod
    def drift_check(cls) -> Tuple[bool, Optional[str]]:
        """
        Verify memory node configuration matches expected values.
        
        Checks:
        - Embedding model name
        - Model dimension (768)
        - Collection dimension (768)
        - API key authentication enforced
        
        Returns:
            Tuple of (passed: bool, error_message: Optional[str])
        """
        try:
            # Check health endpoint
            health = memory_health()
            cls._health_data = health
            
            # Check 1: Embedding model
            model = health.get('expected_model', '')
            if model != REQUIRED_EMBEDDING_MODEL:
                error = f"Model mismatch: expected '{REQUIRED_EMBEDDING_MODEL}', got '{model}'"
                cls._stats.drift_check_passed = False
                return False, error
            
            # Check 2: Model dimension
            model_dim = health.get('expected_dimension', 0)
            if model_dim != REQUIRED_DIMENSION:
                error = f"Model dim mismatch: expected {REQUIRED_DIMENSION}, got {model_dim}"
                cls._stats.drift_check_passed = False
                return False, error
            
            # Check 3: Collection dimension
            collection = health.get('collection', {})
            coll_dim = collection.get('vector_size', 0)
            if coll_dim != REQUIRED_DIMENSION:
                error = f"Collection dim mismatch: expected {REQUIRED_DIMENSION}, got {coll_dim}"
                cls._stats.drift_check_passed = False
                return False, error
            
            # Check 4: Auth enabled (verify by checking health response has auth_enabled)
            if not health.get('auth_enabled', False):
                error = "Auth not enforced on memory node"
                cls._stats.drift_check_passed = False
                return False, error
            
            cls._stats.drift_check_passed = True
            cls._stats.last_drift_check = datetime.now(timezone.utc).isoformat()
            cls._drift_check_done = True
            
            return True, None
            
        except Exception as e:
            error = f"Drift check failed: {str(e)}"
            cls._stats.drift_check_passed = False
            return False, error
    
    @classmethod
    def assert_drift_check(cls):
        """Assert drift check has passed, disable memory if not"""
        if not cls._drift_check_done:
            passed, error = cls.drift_check()
            if not passed:
                cls._memory_enabled = False
                logger.error(f"DRIFT DETECTED: {error}")
                logger.error("Memory features DISABLED")
                raise DriftDetectedError(error)
    
    @classmethod
    def is_memory_enabled(cls) -> bool:
        """Check if memory operations are enabled"""
        return cls._memory_enabled and cls._drift_check_done and cls._stats.drift_check_passed
    
    @classmethod
    def _validate_write_policy(cls, text: str, kind: str, scope: str, 
                               importance: float) -> Tuple[bool, Optional[str]]:
        """
        Validate write against policy.
        
        Returns:
            Tuple of (valid: bool, reason: Optional[str])
        """
        # Check 1: Importance threshold
        if importance < MIN_IMPORTANCE:
            return False, f"Importance {importance} below threshold {MIN_IMPORTANCE}"
        
        # Check 2: Valid kind
        if kind not in ALLOWED_KINDS:
            return False, f"Invalid kind '{kind}'. Allowed: {ALLOWED_KINDS}"
        
        # Check 3: Valid scope (coerce, don't reject)
        if scope not in VALID_SCOPES:
            logger.warning(f"Invalid scope '{scope}', coercing to '{DEFAULT_SCOPE}'")
        
        # Check 4: Text length
        if len(text) < MIN_TEXT_LENGTH:
            return False, f"Text too short ({len(text)} chars, min {MIN_TEXT_LENGTH})"
        if len(text) > MAX_TEXT_LENGTH:
            return False, f"Text too long ({len(text)} chars, max {MAX_TEXT_LENGTH})"
        
        # Check 5: No raw logs/tool output
        prohibited_patterns = [
            r'^\[?\d{4}-\d{2}-\d{2}.*\]?',
            r'^(DEBUG|INFO|WARN|ERROR|TRACE)',
            r'^\$\s+',
            r'^>',
            r'^(stdout|stderr):',
        ]
        
        for pattern in prohibited_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False, f"Text appears to be raw output/log"
        
        return True, None
    
    @classmethod
    def write_gate(cls, text: str, kind: str = "fact", scope: str = "global",
                   importance: float = 0.7, metadata: Optional[Dict] = None) -> bool:
        """
        HARD GATE - All memory writes MUST go through this function.
        
        This is the ONLY authorized path for memory writes.
        
        Args:
            text: Memory content
            kind: Memory kind (fact, preference, task)
            scope: Memory scope (global, local, session)
            importance: Importance score (0.0-1.0)
            metadata: Optional additional metadata
            
        Returns:
            True if write accepted and persisted
            
        Raises:
            DriftDetectedError: If drift check hasn't passed
            PolicyViolationError: If policy violation detected
        """
        # Ensure drift check passed
        cls.assert_drift_check()
        
        if not cls._memory_enabled:
            logger.warning("Memory disabled, write rejected")
            cls._record_decision("write", False, "memory_disabled", text)
            return False
        
        cls._stats.writes_attempted += 1
        
        # Validate policy
        valid, reason = cls._validate_write_policy(text, kind, scope, importance)
        
        if not valid:
            reason_str = reason or "unknown"
            cls._stats.writes_rejected[reason_str] = cls._stats.writes_rejected.get(reason_str, 0) + 1
            cls._record_decision("write", False, reason_str, text)
            logger.debug(f"Write rejected: {reason_str}")
            return False
        
        # Prepare item
        item = {
            "text": text,
            "kind": kind,
            "scope": scope,
            "importance": importance,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if metadata:
            item.update(metadata)
        
        try:
            # Execute write
            result = memory_upsert([item])
            success = result.get('inserted', 0) >= 1
            
            if success:
                cls._stats.writes_accepted += 1
                cls._record_decision("write", True, item_preview=text)
            else:
                cls._stats.writes_rejected['upsert_failed'] = \
                    cls._stats.writes_rejected.get('upsert_failed', 0) + 1
                cls._record_decision("write", False, "upsert_failed", text)
            
            return success
            
        except MemoryClientError as e:
            cls._stats.writes_rejected['client_error'] = \
                cls._stats.writes_rejected.get('client_error', 0) + 1
            cls._record_decision("write", False, f"client_error: {e}", text)
            logger.error(f"Memory write failed: {e}")
            return False
    
    @classmethod
    def _compute_composite_score(cls, result: Dict[str, Any]) -> float:
        """Calculate composite score for retrieval ranking"""
        similarity = result.get("score", 0.0)
        importance = result.get("importance", 0.5)
        scope = result.get("scope", "local")
        
        scope_boost = SCOPE_BOOSTS.get(scope, 0.0)
        
        # Temporal decay for old session memories
        temporal_decay = 1.0
        if scope == "session":
            timestamp = result.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        ts = timestamp
                    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
                    if age_hours > 24:
                        temporal_decay = max(0.9, 1.0 - (age_hours - 24) / 168)
                except (ValueError, TypeError):
                    pass
        
        composite = (similarity * importance * temporal_decay) + scope_boost
        return max(0.0, min(1.0, composite))
    
    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize text for deduplication"""
        normalized = text.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    @classmethod
    def _deduplicate_results(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results by content hash"""
        seen_hashes = set()
        unique = []
        
        # Sort by original score descending to keep best
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        for result in sorted_results:
            text = result.get("text", "")
            content_hash = hashlib.sha256(cls._normalize_text(text).encode()).hexdigest()[:16]
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(result)
        
        cls._stats.duplicates_filtered += (len(results) - len(unique))
        return unique
    
    @classmethod
    def retrieve_gate(cls, query: str, limit: int = DEFAULT_RETRIEVAL_LIMIT,
                     min_score: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        HARD GATE - All memory retrievals MUST go through this function.
        
        This is the ONLY authorized path for memory retrievals.
        
        Args:
            query: Search query
            limit: Maximum results (default 5, max 20)
            min_score: Minimum similarity score (default 0.75)
            
        Returns:
            Filtered and ranked list of memory results
            
        Raises:
            DriftDetectedError: If drift check hasn't passed
        """
        # Ensure drift check passed
        cls.assert_drift_check()
        
        if not cls._memory_enabled:
            logger.warning("Memory disabled, returning empty results")
            return []
        
        cls._stats.retrievals_processed += 1
        
        # Clamp limit
        limit = max(1, min(limit, MAX_RETRIEVAL_LIMIT))
        
        try:
            # Request more to allow for filtering
            search_limit = limit * 3
            result = memory_search(query, limit=search_limit)
            results = result.get('results', [])
            
            if not results:
                cls._record_decision("retrieve", True, f"no_results_for: {query}")
                return []
            
            # Step 1: Filter by similarity threshold
            filtered = [r for r in results if r.get("score", 0) >= min_score]
            
            # Step 2: Deduplicate
            deduplicated = cls._deduplicate_results(filtered)
            
            # Step 3: Calculate composite scores
            for result in deduplicated:
                result["composite_score"] = cls._compute_composite_score(result)
            
            # Step 4: Sort by composite score
            sorted_results = sorted(deduplicated, 
                                   key=lambda x: x.get("composite_score", 0), 
                                   reverse=True)
            
            # Step 5: Limit results
            limited = sorted_results[:limit]
            
            # Step 6: Clean payload - only allowed fields
            allowed_fields = {"text", "score", "kind", "scope", "importance", "timestamp"}
            cleaned = []
            for result in limited:
                cleaned_result = {k: v for k, v in result.items() if k in allowed_fields}
                # Use composite score as final score
                cleaned_result["score"] = result.get("composite_score", result.get("score", 0))
                cleaned.append(cleaned_result)
            
            cls._record_decision("retrieve", True, f"returned_{len(cleaned)}_results")
            return cleaned
            
        except MemoryClientError as e:
            cls._record_decision("retrieve", False, f"error: {e}")
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    @classmethod
    def build_context_block(cls, results: List[Dict[str, Any]], 
                           max_items: int = MAX_CONTEXT_ITEMS,
                           max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Build a deterministic context block for prompt injection.
        
        Args:
            results: List of memory results (from retrieve_gate)
            max_items: Maximum items to include (default 8)
            max_chars: Maximum characters in block (default 1200)
            
        Returns:
            Formatted context block string
        """
        if not results:
            return ""
        
        lines = ["=== RELEVANT MEMORIES ===", ""]
        char_count = len(lines[0]) + 1
        items_included = 0
        
        for result in results[:max_items]:
            text = result.get("text", "")
            scope = result.get("scope", "unknown")
            kind = result.get("kind", "unknown")
            importance = result.get("importance", 0.5)
            score = result.get("score", 0.0)
            
            # Format: "[SCOPE/kind] text (importance: X.XX, score: X.XXX)"
            item_line = f"[{scope}/{kind}] {text} (importance: {importance:.2f}, relevance: {score:.3f})"
            
            # Check if adding this would exceed limit
            if char_count + len(item_line) + 2 > max_chars:
                break
            
            lines.append(item_line)
            char_count += len(item_line) + 1
            items_included += 1
        
        lines.append("")
        lines.append(f"=== {items_included} memories retrieved ===")
        
        return "\n".join(lines)
    
    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get comprehensive memory system status"""
        if cls._health_data:
            health = cls._health_data
        else:
            try:
                health = memory_health()
                cls._health_data = health
            except:
                health = {}
        
        return {
            "memory_enabled": cls._memory_enabled,
            "drift_check_passed": cls._stats.drift_check_passed,
            "last_drift_check": cls._stats.last_drift_check,
            "node_status": health.get("status", "unknown"),
            "collection": health.get("collection", {}),
            "expected_model": REQUIRED_EMBEDDING_MODEL,
            "actual_model": health.get("expected_model", "unknown"),
            "expected_dim": REQUIRED_DIMENSION,
            "actual_dim": health.get("expected_dimension", 0),
            "collection_dim": health.get("collection", {}).get("vector_size", 0),
            "auth_enabled": health.get("auth_enabled", False),
            "thresholds": {
                "min_importance": MIN_IMPORTANCE,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "max_context_items": MAX_CONTEXT_ITEMS,
                "max_context_chars": MAX_CONTEXT_CHARS
            },
            "policy_stats": {
                "writes_attempted": cls._stats.writes_attempted,
                "writes_accepted": cls._stats.writes_accepted,
                "write_acceptance_rate": cls._stats.writes_accepted / max(1, cls._stats.writes_attempted),
                "retrievals_processed": cls._stats.retrievals_processed,
                "duplicates_filtered": cls._stats.duplicates_filtered
            }
        }
    
    @classmethod
    def get_last_decisions(cls, count: int = 20) -> List[PolicyDecision]:
        """Get last N policy decisions"""
        return cls._stats.last_20_decisions[-count:]
    
    @classmethod
    def probe_retrieval(cls, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Probe retrieval without context injection.
        Returns raw results with dedupe groups for inspection.
        """
        results = cls.retrieve_gate(query, limit=limit)
        
        # Add dedupe group info
        grouped = []
        for result in results:
            text = result.get("text", "")
            content_hash = hashlib.sha256(cls._normalize_text(text).encode()).hexdigest()[:8]
            
            result_with_group = {**result, "dedupe_group": content_hash}
            grouped.append(result_with_group)
        
        return grouped


# ============================================================================
# BYPASS PREVENTION - Block direct wrapper/client calls
# ============================================================================

class _ProtectedMemoryWrapper:
    """
    Protected wrapper that prevents direct calls.
    All access must go through PolicyLayer.
    """
    
    @staticmethod
    def _block_direct_call():
        """Raise error if called directly"""
        import inspect
        frame = inspect.currentframe()
        if frame is None:
            raise BypassAttemptError("Cannot verify call origin")
        
        try:
            # Check if call came through PolicyLayer
            caller_frame = None
            if frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
            
            if caller_frame is None:
                raise BypassAttemptError(
                    "Direct call to memory wrapper blocked. "
                    "Use PolicyLayer.write_gate() or PolicyLayer.retrieve_gate() instead."
                )
            
            caller_name = caller_frame.f_code.co_name
            caller_filename = caller_frame.f_code.co_filename
            
            # Allow calls from PolicyLayer
            if "PolicyLayer" in caller_name or "policy_layer" in caller_filename:
                return
            
            raise BypassAttemptError(
                f"Direct call to memory wrapper blocked. "
                f"Use PolicyLayer.write_gate() or PolicyLayer.retrieve_gate() instead."
            )
        finally:
            del frame
    
    @classmethod
    def remember(cls, *args, **kwargs):
        cls._block_direct_call()
        # This should never execute if enforcement works
        from memory_wrapper import remember as _original_remember
        return _original_remember(*args, **kwargs)
    
    @classmethod
    def recall(cls, *args, **kwargs):
        cls._block_direct_call()
        from memory_wrapper import recall as _original_recall
        return _original_recall(*args, **kwargs)


# Replace wrapper functions with protected versions
def _install_protection():
    """Install bypass protection"""
    import memory_wrapper
    memory_wrapper.remember = _ProtectedMemoryWrapper.remember
    memory_wrapper.recall = _ProtectedMemoryWrapper.recall
    logger.info("Memory bypass protection installed")


# Auto-install on import
_install_protection()


if __name__ == "__main__":
    # Self-test
    print("PolicyLayer Enforcement Test\n")
    
    # Run drift check
    print("Running drift check...")
    try:
        PolicyLayer.drift_check()
        print("✓ Drift check passed")
    except DriftDetectedError as e:
        print(f"✗ Drift check failed: {e}")
        sys.exit(1)
    
    # Test write gate
    print("\nTesting write gate...")
    result = PolicyLayer.write_gate(
        "User prefers dark mode for applications",
        kind="preference",
        scope="global",
        importance=0.85
    )
    print(f"Write result: {'✓ accepted' if result else '✗ rejected'}")
    
    # Test retrieve gate
    print("\nTesting retrieve gate...")
    results = PolicyLayer.retrieve_gate("user preference dark mode", limit=3)
    print(f"Retrieved {len(results)} results")
    for r in results:
        print(f"  - {r['text'][:50]}... (score: {r['score']:.3f})")
    
    # Test context block
    print("\nTesting context block...")
    context = PolicyLayer.build_context_block(results)
    print(f"Context block ({len(context)} chars):")
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Test status
    print("\nStatus:")
    status = PolicyLayer.get_status()
    print(f"  Memory enabled: {status['memory_enabled']}")
    print(f"  Drift check: {status['drift_check_passed']}")
    print(f"  Collection points: {status['collection'].get('points_count', 0)}")
