#!/usr/bin/env python3
"""
Architecture Introspection Layer (Proprioception)

Runtime architecture awareness for the agent.
Automatically detects system topology at startup without manual configuration.
This is runtime-only awareness - no architecture facts are stored in long-term memory.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_client import memory_health, MemoryClientError
from memory_controller_config import REQUIRED_EMBEDDING_MODEL, REQUIRED_DIMENSION


# Configuration
ARCHITECTURE_LOG_PATH = "logs/architecture_probe.log"
ARCHITECTURE_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB
ARCHITECTURE_LOG_BACKUP_COUNT = 3
CACHE_TTL_SECONDS = 600  # 10 minutes


# Setup logging
logger = logging.getLogger("architecture_probe")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    log_dir = os.path.dirname(ARCHITECTURE_LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    handler = RotatingFileHandler(
        ARCHITECTURE_LOG_PATH,
        maxBytes=ARCHITECTURE_LOG_MAX_BYTES,
        backupCount=ARCHITECTURE_LOG_BACKUP_COUNT
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class ArchitectureSnapshot:
    """Snapshot of system architecture state"""
    timestamp: str
    memory_reachable: bool
    qdrant_version: Optional[str]
    collection_name: Optional[str]
    vector_dim: Optional[int]
    embedding_model: Optional[str]
    points_count: Optional[int]
    policy_layer_active: bool
    memory_controller_active: bool
    mode_router_active: bool
    drift_detected: bool
    memory_disabled_reason: Optional[str] = None


class ArchitectureIntrospector:
    """
    Runtime architecture introspection layer.
    
    Automatically detects system topology without manual configuration.
    Provides proprioception - the agent knows its own architecture.
    """
    
    _instance = None
    _cache: Optional[ArchitectureSnapshot] = None
    _cache_timestamp: Optional[float] = None
    _memory_disabled: bool = False
    _disable_reason: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _is_cache_valid(self) -> bool:
        """Check if cached snapshot is still valid"""
        if self._cache is None or self._cache_timestamp is None:
            return False
        
        age = time.time() - self._cache_timestamp
        return age < CACHE_TTL_SECONDS
    
    def probe_memory_node(self) -> Dict[str, Any]:
        """
        Probe the memory node health endpoint.
        
        Returns:
            Dict with:
            - memory_reachable: bool
            - qdrant_version: str or None
            - collection: str or None
            - vector_dim: int or None
            - embedding_model: str or None
            - points_count: int or None
            - error: str or None
        """
        result = {
            "memory_reachable": False,
            "qdrant_version": None,
            "collection": None,
            "vector_dim": None,
            "embedding_model": None,
            "points_count": None,
            "error": None
        }
        
        try:
            health = memory_health()
            result["memory_reachable"] = True
            result["qdrant_version"] = health.get("qdrant_version", "unknown")
            result["embedding_model"] = health.get("expected_model", "unknown")
            result["vector_dim"] = health.get("expected_dimension")
            
            collection = health.get("collection", {})
            result["collection"] = collection.get("name", "unknown")
            result["points_count"] = collection.get("points_count", 0)
            
            logger.info(f"Memory node probe successful: {result['qdrant_version']}")
            
        except MemoryClientError as e:
            result["error"] = f"MemoryClientError: {str(e)}"
            logger.error(f"Memory node unreachable: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"Memory node probe failed: {e}")
        
        return result
    
    def probe_local_system(self) -> Dict[str, bool]:
        """
        Probe local system components.
        
        Returns:
            Dict with:
            - policy_layer_active: bool
            - memory_controller_active: bool
            - mode_router_active: bool
        """
        result = {
            "policy_layer_active": False,
            "memory_controller_active": False,
            "mode_router_active": False
        }
        
        # Check PolicyLayer
        try:
            from policy_layer_enforcement import PolicyLayer
            # Try to get status - if it works, it's active
            status = PolicyLayer.get_status()
            result["policy_layer_active"] = status.get("healthy", False)
        except Exception as e:
            logger.debug(f"PolicyLayer not active: {e}")
        
        # Check MemoryController
        try:
            from memory_controller import MemoryController
            mc = MemoryController()
            result["memory_controller_active"] = mc.is_memory_enabled()
        except Exception as e:
            logger.debug(f"MemoryController not active: {e}")
        
        # Check ModeRouter
        try:
            from mode_router import ModeRouter
            router = ModeRouter()
            # ModeRouter is considered active if it can be instantiated
            result["mode_router_active"] = True
        except Exception as e:
            logger.debug(f"ModeRouter not active: {e}")
        
        return result
    
    def check_drift(self, probe_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check for architecture drift.
        
        Returns:
            Tuple of (drift_detected, reason)
        """
        if not probe_result["memory_reachable"]:
            return False, None  # Not drift, just unreachable
        
        # Check vector dimension
        actual_dim = probe_result.get("vector_dim")
        if actual_dim is not None and actual_dim != REQUIRED_DIMENSION:
            return True, f"Dimension mismatch: expected {REQUIRED_DIMENSION}, got {actual_dim}"
        
        # Check embedding model
        actual_model = probe_result.get("embedding_model")
        if actual_model and actual_model != REQUIRED_EMBEDDING_MODEL:
            return True, f"Model mismatch: expected {REQUIRED_EMBEDDING_MODEL}, got {actual_model}"
        
        return False, None
    
    def get_snapshot(self, force_refresh: bool = False) -> ArchitectureSnapshot:
        """
        Get current architecture snapshot.
        
        Args:
            force_refresh: If True, ignore cache and re-probe
            
        Returns:
            ArchitectureSnapshot with current state
        """
        # Return cached snapshot if valid and not forcing refresh
        if not force_refresh and self._is_cache_valid():
            return self._cache
        
        # Probe memory node
        memory_probe = self.probe_memory_node()
        
        # Probe local system
        local_probe = self.probe_local_system()
        
        # Check for drift
        drift_detected, drift_reason = self.check_drift(memory_probe)
        
        # Determine memory status
        memory_disabled_reason = None
        if not memory_probe["memory_reachable"]:
            memory_disabled_reason = "Memory node unreachable"
            self._memory_disabled = True
            self._disable_reason = memory_disabled_reason
        elif drift_detected:
            memory_disabled_reason = f"Drift detected: {drift_reason}"
            self._memory_disabled = True
            self._disable_reason = memory_disabled_reason
            logger.error(f"ARCHITECTURE DRIFT: {drift_reason}")
        else:
            self._memory_disabled = False
            self._disable_reason = None
        
        # Build snapshot
        snapshot = ArchitectureSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            memory_reachable=memory_probe["memory_reachable"],
            qdrant_version=memory_probe["qdrant_version"],
            collection_name=memory_probe["collection"],
            vector_dim=memory_probe["vector_dim"],
            embedding_model=memory_probe["embedding_model"],
            points_count=memory_probe["points_count"],
            policy_layer_active=local_probe["policy_layer_active"],
            memory_controller_active=local_probe["memory_controller_active"],
            mode_router_active=local_probe["mode_router_active"],
            drift_detected=drift_detected,
            memory_disabled_reason=memory_disabled_reason
        )
        
        # Update cache
        self._cache = snapshot
        self._cache_timestamp = time.time()
        
        # Log snapshot
        logger.info(f"Architecture snapshot: {json.dumps(asdict(snapshot))}")
        
        return snapshot
    
    def build_system_context_block(self, force_refresh: bool = False) -> str:
        """
        Build system context block for injection into agent context.
        
        This is runtime-only and is NOT stored in long-term memory.
        
        Args:
            force_refresh: If True, get fresh snapshot
            
        Returns:
            Formatted context block string
        """
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        
        lines = [
            "=" * 60,
            "SYSTEM ARCHITECTURE SNAPSHOT (Runtime Only)",
            "=" * 60,
            "",
            f"Timestamp: {snapshot.timestamp}",
            "",
            "Memory Infrastructure:",
            f"  Reachable: {'✓ YES' if snapshot.memory_reachable else '✗ NO'}",
        ]
        
        if snapshot.memory_reachable:
            lines.extend([
                f"  Qdrant Version: {snapshot.qdrant_version or 'unknown'}",
                f"  Collection: {snapshot.collection_name or 'unknown'}",
                f"  Vector Dimension: {snapshot.vector_dim or 'unknown'}",
                f"  Embedding Model: {snapshot.embedding_model or 'unknown'}",
                f"  Points Stored: {snapshot.points_count or 0:,}",
            ])
        
        if snapshot.memory_disabled_reason:
            lines.extend([
                "",
                f"⚠ Memory Disabled: {snapshot.memory_disabled_reason}",
            ])
        
        if snapshot.drift_detected:
            lines.extend([
                "",
                "⚠ ARCHITECTURE DRIFT DETECTED",
                f"  Reason: {snapshot.memory_disabled_reason}",
            ])
        
        lines.extend([
            "",
            "Local Components:",
            f"  Policy Layer: {'✓ Active' if snapshot.policy_layer_active else '✗ Inactive'}",
            f"  Memory Controller: {'✓ Active' if snapshot.memory_controller_active else '✗ Inactive'}",
            f"  Mode Router: {'✓ Active' if snapshot.mode_router_active else '✗ Inactive'}",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def is_memory_usable(self) -> bool:
        """Check if memory system is usable"""
        if not self._is_cache_valid():
            self.get_snapshot()
        return not self._memory_disabled
    
    def get_memory_disable_reason(self) -> Optional[str]:
        """Get reason memory is disabled, if any"""
        return self._disable_reason
    
    def invalidate_cache(self):
        """Force cache invalidation"""
        self._cache = None
        self._cache_timestamp = None
        logger.info("Architecture cache invalidated")
    
    def get_cached_snapshot(self) -> Optional[ArchitectureSnapshot]:
        """Get cached snapshot if valid"""
        if self._is_cache_valid():
            return self._cache
        return None


# Global instance
_introspector: Optional[ArchitectureIntrospector] = None


def get_introspector() -> ArchitectureIntrospector:
    """Get singleton introspector instance"""
    global _introspector
    if _introspector is None:
        _introspector = ArchitectureIntrospector()
    return _introspector


def probe_memory_node() -> Dict[str, Any]:
    """Convenience function to probe memory node"""
    return get_introspector().probe_memory_node()


def probe_local_system() -> Dict[str, bool]:
    """Convenience function to probe local system"""
    return get_introspector().probe_local_system()


def build_system_context_block(force_refresh: bool = False) -> str:
    """Convenience function to build context block"""
    return get_introspector().build_system_context_block(force_refresh=force_refresh)


def is_memory_usable() -> bool:
    """Convenience function to check memory usability"""
    return get_introspector().is_memory_usable()


def check_drift() -> Tuple[bool, Optional[str]]:
    """Check for architecture drift"""
    introspector = get_introspector()
    snapshot = introspector.get_snapshot(force_refresh=True)
    return snapshot.drift_detected, snapshot.memory_disabled_reason


if __name__ == "__main__":
    # Self-test
    print("Architecture Introspection Self-Test\n")
    
    introspector = get_introspector()
    
    print("1. Probing memory node...")
    memory_probe = introspector.probe_memory_node()
    print(f"   Reachable: {memory_probe['memory_reachable']}")
    if memory_probe['memory_reachable']:
        print(f"   Qdrant: {memory_probe['qdrant_version']}")
        print(f"   Model: {memory_probe['embedding_model']}")
        print(f"   Dimension: {memory_probe['vector_dim']}")
    
    print("\n2. Probing local system...")
    local_probe = introspector.probe_local_system()
    print(f"   Policy Layer: {local_probe['policy_layer_active']}")
    print(f"   Memory Controller: {local_probe['memory_controller_active']}")
    print(f"   Mode Router: {local_probe['mode_router_active']}")
    
    print("\n3. Building system context block...")
    context = introspector.build_system_context_block()
    print(context)
    
    print("\n4. Checking drift...")
    drift, reason = introspector.check_drift(memory_probe)
    print(f"   Drift detected: {drift}")
    if drift:
        print(f"   Reason: {reason}")
    
    print("\n5. Memory usable:", introspector.is_memory_usable())
    
    print("\n✓ Self-test complete")
