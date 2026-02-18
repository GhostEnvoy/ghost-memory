#!/usr/bin/env python3
"""
Memory Controller - Prefrontal Orchestration Layer

This is the top-level orchestration layer that governs WHEN memory is accessed,
not just HOW it is validated. All memory operations must route through this controller.

Architecture:
    User → Agent → MemoryController → PolicyLayer → Memory API

Usage:
    from memory_controller import MemoryController
    
    controller = MemoryController(session_id="abc123")
    
    # Controlled retrieval
    if controller.should_retrieve("what are my preferences"):
        results = controller.controlled_retrieve("preferences", limit=5)
        context = controller.build_context(results)
    
    # Controlled write
    controller.controlled_write(
        text="User prefers dark mode",
        kind="preference",
        scope="global",
        importance=0.8,
        justification="Explicitly stated preference"
    )
"""

import os
import sys
import hashlib
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field, asdict

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_layer_enforcement import PolicyLayer, DriftDetectedError
from memory_controller_config import (
    MAX_WRITES_PER_SESSION,
    RETRIEVAL_ENABLED,
    WRITE_ENABLED,
    MIN_IMPORTANCE,
    AUDIT_LOG_PATH,
    AUDIT_LOG_MAX_BYTES,
    AUDIT_LOG_BACKUP_COUNT,
    REQUIRED_EMBEDDING_MODEL,
    REQUIRED_DIMENSION,
    RETRIEVAL_TRIGGERS,
    RETRIEVAL_EXCLUSIONS,
    MAX_CONTEXT_ITEMS,
    MAX_CONTEXT_CHARS,
)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class MemoryBypassError(Exception):
    """Raised when attempting to bypass MemoryController"""
    pass


class MemoryWriteLimitExceeded(Exception):
    """Raised when session write limit is exceeded"""
    pass


class DuplicateWriteError(Exception):
    """Raised when attempting to write duplicate content"""
    pass


class MemoryDisabledError(Exception):
    """Raised when memory system is disabled"""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievalDecision:
    """Record of a retrieval decision"""
    timestamp: str
    query: str
    allowed: bool
    reason: str
    session_id: str


@dataclass
class WriteDecision:
    """Record of a write decision"""
    timestamp: str
    text_hash: str
    kind: str
    scope: str
    importance: float
    allowed: bool
    reason: str
    session_id: str
    justification: Optional[str] = None


@dataclass
class SessionState:
    """Track session memory usage"""
    session_id: str
    write_count: int = 0
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    written_hashes: set = field(default_factory=set)


# ============================================================================
# AUDIT LOGGER SETUP
# ============================================================================

def setup_audit_logger() -> logging.Logger:
    """Setup rotating audit logger"""
    logger = logging.getLogger("memory_audit")
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers = []
    
    # Ensure logs directory exists
    log_dir = os.path.dirname(AUDIT_LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create rotating file handler
    handler = RotatingFileHandler(
        AUDIT_LOG_PATH,
        maxBytes=AUDIT_LOG_MAX_BYTES,
        backupCount=AUDIT_LOG_BACKUP_COUNT
    )
    
    # Simple formatter - one JSON line per entry
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Global audit logger
_audit_logger = setup_audit_logger()


# ============================================================================
# MEMORY CONTROLLER
# ============================================================================

class MemoryController:
    """
    Orchestration layer for memory operations.
    
    This is the ONLY entry point for memory operations.
    All calls must go through this controller.
    """
    
    _instance = None
    _initialized = False
    _drift_verified = False
    _memory_enabled = True
    _session_state: Optional[SessionState] = None
    _retrieval_history: List[RetrievalDecision] = field(default_factory=list)
    _write_history: List[WriteDecision] = field(default_factory=list)
    
    def __new__(cls, session_id: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, session_id: Optional[str] = None):
        if not self._initialized:
            self._initialized = True
            self._retrieval_history = []
            self._write_history = []
            
            # Initialize or update session
            if session_id:
                self._initialize_session(session_id)
            
            # Run drift check on first initialization
            if not self._drift_verified:
                self._verify_drift_protection()
    
    def _initialize_session(self, session_id: str):
        """Initialize or validate session state"""
        if (self._session_state is None or 
            self._session_state.session_id != session_id):
            # New session
            self._session_state = SessionState(session_id=session_id)
            self._log_audit("session_start", {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        else:
            # Update activity timestamp
            self._session_state.last_activity = datetime.now(timezone.utc).isoformat()
    
    def _verify_drift_protection(self):
        """Verify memory node configuration on startup"""
        try:
            # Get health data from PolicyLayer
            status = PolicyLayer.get_status()
            
            checks = []
            
            # Check 1: Model name
            actual_model = status.get("actual_model", "")
            checks.append(("model", actual_model == REQUIRED_EMBEDDING_MODEL))
            
            # Check 2: Model dimension
            actual_dim = status.get("actual_dim", 0)
            checks.append(("model_dim", actual_dim == REQUIRED_DIMENSION))
            
            # Check 3: Collection dimension
            collection_dim = status.get("collection_dim", 0)
            checks.append(("collection_dim", collection_dim == REQUIRED_DIMENSION))
            
            # All checks must pass
            all_passed = all(passed for _, passed in checks)
            
            if not all_passed:
                failed = [name for name, passed in checks if not passed]
                MemoryController._memory_enabled = False
                self._log_audit("drift_detected", {
                    "failed_checks": failed,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                raise DriftDetectedError(f"Drift detected in: {', '.join(failed)}")
            
            self._drift_verified = True
            self._log_audit("drift_check_passed", {
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            MemoryController._memory_enabled = False
            self._log_audit("drift_check_failed", {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise DriftDetectedError(f"Drift verification failed: {e}")
    
    def _log_audit(self, action_type: str, data: Dict[str, Any]):
        """Log action to audit log"""
        entry = {
            "type": action_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data
        }
        _audit_logger.info(json.dumps(entry))
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute hash for text deduplication"""
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]
    
    # ========================================================================
    # RETRIEVAL GATE
    # ========================================================================
    
    def should_retrieve(self, query: str) -> Tuple[bool, str]:
        """
        Decide if retrieval should be allowed for this query.
        
        Rules:
        - Allow for long-term knowledge, identity, preferences, context continuity
        - Deny for casual chat, ephemeral remarks, tool-only interactions
        
        Args:
            query: The search query
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if not RETRIEVAL_ENABLED:
            return False, "Retrieval disabled"
        
        if not self._memory_enabled:
            return False, "Memory system disabled"
        
        if not query or not query.strip():
            return False, "Empty query"
        
        query_lower = query.lower().strip()
        
        # Check exclusion patterns first
        for exclusion in RETRIEVAL_EXCLUSIONS:
            if exclusion in query_lower:
                return False, f"Casual/ephemeral pattern detected: '{exclusion}'"
        
        # Check trigger patterns
        for trigger in RETRIEVAL_TRIGGERS:
            if trigger in query_lower:
                return True, f"Retrieval trigger matched: '{trigger}'"
        
        # Default: allow retrieval but flag for review
        return True, "No strong indicators, allowing with caution"
    
    # ========================================================================
    # WRITE GATE
    # ========================================================================
    
    def controlled_write(self, text: str, kind: str, scope: str, 
                        importance: float, justification: Optional[str] = None) -> bool:
        """
        Controlled memory write with session limits and deduplication.
        
        Args:
            text: Memory content
            kind: Memory kind (fact, preference, task)
            scope: Memory scope (global, local, session)
            importance: Importance score (0.0-1.0)
            justification: Why this memory should be stored
            
        Returns:
            True if write succeeded
            
        Raises:
            MemoryWriteLimitExceeded: If session write limit exceeded
            DuplicateWriteError: If duplicate detected
            MemoryDisabledError: If memory system disabled
        """
        if not WRITE_ENABLED:
            raise MemoryDisabledError("Write operations disabled")
        
        if not self._memory_enabled:
            raise MemoryDisabledError("Memory system disabled due to drift")
        
        if not self._session_state:
            raise MemoryDisabledError("No active session")
        
        # Check session write limit
        if self._session_state.write_count >= MAX_WRITES_PER_SESSION:
            self._log_audit("write_rejected", {
                "reason": "session_limit_exceeded",
                "session_id": self._session_state.session_id,
                "current_count": self._session_state.write_count,
                "limit": MAX_WRITES_PER_SESSION
            })
            raise MemoryWriteLimitExceeded(
                f"Session write limit exceeded ({MAX_WRITES_PER_SESSION} writes per session)"
            )
        
        # Compute text hash
        text_hash = self._compute_text_hash(text)
        
        # Check for duplicates in session
        if text_hash in self._session_state.written_hashes:
            self._log_audit("write_rejected", {
                "reason": "duplicate_in_session",
                "text_hash": text_hash,
                "session_id": self._session_state.session_id
            })
            raise DuplicateWriteError("Duplicate content detected in current session")
        
        # Call PolicyLayer write gate
        try:
            success = PolicyLayer.write_gate(
                text=text,
                kind=kind,
                scope=scope,
                importance=importance
            )
            
            if success:
                # Update session state
                self._session_state.write_count += 1
                self._session_state.written_hashes.add(text_hash)
                self._session_state.last_activity = datetime.now(timezone.utc).isoformat()
                
                # Log success
                self._log_audit("write_accepted", {
                    "text_hash": text_hash,
                    "kind": kind,
                    "scope": scope,
                    "importance": importance,
                    "justification": justification,
                    "session_id": self._session_state.session_id,
                    "write_number": self._session_state.write_count
                })
                
                # Record decision
                decision = WriteDecision(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    text_hash=text_hash,
                    kind=kind,
                    scope=scope,
                    importance=importance,
                    allowed=True,
                    reason="Policy and session checks passed",
                    session_id=self._session_state.session_id,
                    justification=justification
                )
                self._write_history.append(decision)
                
            else:
                # PolicyLayer rejected
                self._log_audit("write_rejected", {
                    "text_hash": text_hash,
                    "kind": kind,
                    "scope": scope,
                    "importance": importance,
                    "reason": "policy_layer_rejected",
                    "session_id": self._session_state.session_id
                })
                
                decision = WriteDecision(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    text_hash=text_hash,
                    kind=kind,
                    scope=scope,
                    importance=importance,
                    allowed=False,
                    reason="PolicyLayer rejection",
                    session_id=self._session_state.session_id,
                    justification=justification
                )
                self._write_history.append(decision)
            
            return success
            
        except Exception as e:
            self._log_audit("write_error", {
                "text_hash": text_hash,
                "error": str(e),
                "session_id": self._session_state.session_id
            })
            raise
    
    # ========================================================================
    # RETRIEVAL WRAPPER
    # ========================================================================
    
    def controlled_retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Controlled memory retrieval with gate check.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of memory results (empty if retrieval not allowed)
        """
        # Check if retrieval allowed
        allowed, reason = self.should_retrieve(query)
        
        # Log decision
        decision = RetrievalDecision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            allowed=allowed,
            reason=reason,
            session_id=self._session_state.session_id if self._session_state else "none"
        )
        self._retrieval_history.append(decision)
        
        self._log_audit("retrieval_decision", {
            "query": query,
            "allowed": allowed,
            "reason": reason,
            "session_id": self._session_state.session_id if self._session_state else "none"
        })
        
        if not allowed:
            return []
        
        # Perform retrieval through PolicyLayer
        try:
            results = PolicyLayer.retrieve_gate(query, limit=limit)
            
            self._log_audit("retrieval_success", {
                "query": query,
                "results_count": len(results),
                "session_id": self._session_state.session_id if self._session_state else "none"
            })
            
            return results
            
        except Exception as e:
            self._log_audit("retrieval_error", {
                "query": query,
                "error": str(e),
                "session_id": self._session_state.session_id if self._session_state else "none"
            })
            return []
    
    # ========================================================================
    # CONTEXT BUILDER
    # ========================================================================
    
    def build_context(self, query: str) -> str:
        """
        Build context block for prompt injection.
        
        Only builds context if retrieval is allowed.
        
        Args:
            query: Query to retrieve memories for
            
        Returns:
            Formatted context block (empty if retrieval not allowed)
        """
        # Check if retrieval allowed
        allowed, reason = self.should_retrieve(query)
        
        if not allowed:
            return ""
        
        # Retrieve memories
        results = self.controlled_retrieve(query, limit=MAX_CONTEXT_ITEMS)
        
        if not results:
            return ""
        
        # Build context block using PolicyLayer
        return PolicyLayer.build_context_block(results, 
                                               max_items=MAX_CONTEXT_ITEMS,
                                               max_chars=MAX_CONTEXT_CHARS)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        if not self._session_state:
            return {"error": "No active session"}
        
        return {
            "session_id": self._session_state.session_id,
            "write_count": self._session_state.write_count,
            "write_limit": MAX_WRITES_PER_SESSION,
            "writes_remaining": MAX_WRITES_PER_SESSION - self._session_state.write_count,
            "last_activity": self._session_state.last_activity,
            "memory_enabled": self._memory_enabled,
            "drift_verified": self._drift_verified
        }
    
    def get_retrieval_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent retrieval decisions"""
        return [asdict(d) for d in self._retrieval_history[-limit:]]
    
    def get_write_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent write decisions"""
        return [asdict(d) for d in self._write_history[-limit:]]
    
    def is_memory_enabled(self) -> bool:
        """Check if memory system is enabled"""
        return self._memory_enabled and self._drift_verified


# ============================================================================
# BYPASS PROTECTION
# ============================================================================

def _protect_policy_layer():
    """
    Protect PolicyLayer from direct access.
    All access must go through MemoryController.
    """
    # Store original methods
    original_write_gate = PolicyLayer.write_gate
    original_retrieve_gate = PolicyLayer.retrieve_gate
    
    def protected_write_gate(*args, **kwargs):
        import inspect
        # Check if call came from MemoryController
        frame = inspect.currentframe()
        try:
            caller_found = False
            # Walk up the stack
            while frame:
                if frame.f_code.co_name == "controlled_write" and \
                   "memory_controller" in frame.f_code.co_filename:
                    caller_found = True
                    break
                frame = frame.f_back
            
            if not caller_found:
                raise MemoryBypassError(
                    "Direct PolicyLayer access blocked. Use MemoryController instead."
                )
            
            return original_write_gate(*args, **kwargs)
        finally:
            if frame:
                del frame
    
    def protected_retrieve_gate(*args, **kwargs):
        import inspect
        # Check if call came from MemoryController
        frame = inspect.currentframe()
        try:
            caller_found = False
            # Walk up the stack
            while frame:
                if frame.f_code.co_name == "controlled_retrieve" and \
                   "memory_controller" in frame.f_code.co_filename:
                    caller_found = True
                    break
                frame = frame.f_back
            
            if not caller_found:
                raise MemoryBypassError(
                    "Direct PolicyLayer access blocked. Use MemoryController instead."
                )
            
            return original_retrieve_gate(*args, **kwargs)
        finally:
            if frame:
                del frame
    
    # Replace methods
    PolicyLayer.write_gate = staticmethod(protected_write_gate)
    PolicyLayer.retrieve_gate = staticmethod(protected_retrieve_gate)


# Install protection on import
_protect_policy_layer()


if __name__ == "__main__":
    # Self-test
    print("Memory Controller Self-Test\n")
    
    try:
        controller = MemoryController(session_id="test_session_001")
        print("✓ Controller initialized")
        print(f"✓ Memory enabled: {controller.is_memory_enabled()}")
        
        # Test retrieval decision
        print("\nTesting retrieval decisions:")
        queries = [
            "what are my preferences",
            "hello",
            "remember my setup",
            "run this command"
        ]
        
        for query in queries:
            allowed, reason = controller.should_retrieve(query)
            status = "ALLOW" if allowed else "DENY"
            print(f"  [{status}] '{query}' → {reason}")
        
        # Test controlled retrieval
        print("\nTesting controlled retrieval:")
        results = controller.controlled_retrieve("user preferences", limit=3)
        print(f"  Retrieved {len(results)} results")
        
        # Test context building
        print("\nTesting context building:")
        context = controller.build_context("user preferences")
        print(f"  Context length: {len(context)} chars")
        
        # Test session stats
        print("\nSession stats:")
        stats = controller.get_session_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✓ All self-tests passed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
