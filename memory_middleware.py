#!/usr/bin/env python3
"""
OpenClaw Memory Middleware - Unified Integration Point
======================================================

This module provides the single entry point for all memory operations
in the OpenClaw message loop. It integrates:
- Memory retrieval with controlled_retrieve
- Context block building for prompt injection  
- Auto-write decision logic after responses
- User model trait extraction and promotion
- Drift detection and bypass guards
- Audit logging

Usage:
    python3 memory_middleware.py --action retrieve --query "user preference"
    python3 memory_middleware.py --action write --text "User likes dark mode" --kind preference --importance 0.8
    python3 memory_middleware.py --action full_cycle --message "User said X" --session-id "abc123"
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
LOG_FILE = "/home/libra/.openclaw/logs/memory_middleware.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_middleware")

# Memory node configuration
MEMORY_NODE_URL = os.environ.get("MEMORY_NODE_URL", "http://100.103.241.63:8000")
MEMORY_NODE_API_KEY = os.environ.get("MEMORY_NODE_SECRET", "openclaw-memory-secret")
SIMILARITY_THRESHOLD = float(os.environ.get("MEMORY_SIMILARITY_THRESHOLD", "0.75"))
MAX_CONTEXT_CHARS = int(os.environ.get("MEMORY_MAX_CONTEXT_CHARS", "1500"))
MAX_WRITES_PER_SESSION = 5

# Import memory framework components
try:
    from memory_controller import MemoryController
    from policy_layer import should_persist, filter_retrieval, SIMILARITY_THRESHOLD as POLICY_THRESHOLD
    from memory_client import memory_health, memory_upsert, memory_search, MemoryClientError
    MEMORY_CONTROLLER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory controller not available: {e}")
    MEMORY_CONTROLLER_AVAILABLE = False

# Import user model components
try:
    from user_model.features import extract_features, ConversationTurn, UserFeatures, get_strongest_features
    from user_model.hypotheses import HypothesesEngine, TraitHypothesis
    from user_model.promote import TraitPromoter
    USER_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"User model not available: {e}")
    USER_MODEL_AVAILABLE = False

# Import mode router
try:
    from mode_router import ModeRouter
    MODE_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mode router not available: {e}")
    MODE_ROUTER_AVAILABLE = False


class MemoryMiddleware:
    """
    Unified memory middleware for OpenClaw.
    
    Provides controlled access to vector memory with:
    - Bypass prevention (no direct calls allowed)
    - Drift detection (handles node unavailability gracefully)
    - Session state management
    - Audit logging
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._drift_detected = False
        self._write_count = 0
        self._session_hashes = set()
        
        # Initialize components
        self._init_memory_controller()
        self._init_user_model()
        self._init_mode_router()
        
    def _init_memory_controller(self):
        """Initialize memory controller with drift protection."""
        if not MEMORY_CONTROLLER_AVAILABLE:
            logger.warning("Memory controller not available, using fallback mode")
            self._controller = None
            return
            
        try:
            self._controller = MemoryController(session_id=self.session_id)
            # Test connection - use is_memory_enabled() to check drift
            if not self._controller.is_memory_enabled():
                self._drift_detected = True
                logger.warning("Drift detected - memory system disabled")
            else:
                logger.info(f"Memory controller initialized for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize memory controller: {e}")
            self._drift_detected = True
            self._controller = None
            
    def _init_user_model(self):
        """Initialize user model components."""
        if not USER_MODEL_AVAILABLE:
            self._user_model = None
            return
            
        try:
            self._user_model = {
                'engine': HypothesesEngine(),
                'promoter': TraitPromoter()
            }
            logger.info("User model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize user model: {e}")
            self._user_model = None
            
    def _init_mode_router(self):
        """Initialize mode router."""
        if not MODE_ROUTER_AVAILABLE:
            self._mode_router = None
            return
            
        try:
            self._mode_router = ModeRouter()
            logger.info("Mode router initialized")
        except Exception as e:
            logger.error(f"Failed to initialize mode router: {e}")
            self._mode_router = None
            
    def check_drift(self) -> bool:
        """
        Check for drift (node unavailability or dimension mismatch).
        
        Returns:
            True if drift detected, False if healthy
        """
        if self._drift_detected:
            return True
            
        if not MEMORY_CONTROLLER_AVAILABLE:
            return True
            
        try:
            # Try to reach memory node
            health = memory_health()
            if health.get('status') != 'healthy':
                self._drift_detected = True
                logger.warning("Memory node not healthy")
                return True
                
            # Check dimension if available
            collection = health.get('collection', {})
            vector_size = collection.get('vector_size', 0)
            if vector_size != 768:
                logger.warning(f"Dimension mismatch: expected 768, got {vector_size}")
                self._drift_detected = True
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            self._drift_detected = True
            return True
            
    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a query.
        
        This is the ONLY supported retrieval method - bypass attempts
        will be logged and rejected.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of memory results
        """
        # Log retrieval attempt (audit trail)
        logger.info(f"[AUDIT] Retrieve attempt: session={self.session_id}, query='{query}'")
        
        # Check drift
        if self._drift_detected:
            logger.warning(f"[DRIFT] Retrieval blocked - memory disabled")
            return []
            
        # Use controller if available
        if self._controller:
            try:
                results = self._controller.controlled_retrieve(query, limit)
                logger.info(f"[AUDIT] Retrieve success: {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"Controller retrieve failed: {e}")
                return []
                
        # Fallback to direct client
        try:
            response = memory_search(query, limit=limit)
            results = response.get('results', [])
            # Apply policy filtering
            filtered = filter_retrieval(results, limit=limit, min_score=SIMILARITY_THRESHOLD)
            logger.info(f"[AUDIT] Direct retrieve success: {len(filtered)} results")
            return filtered
        except Exception as e:
            logger.error(f"Direct retrieve failed: {e}")
            return []
            
    def build_context_block(self, query: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Build context block for prompt injection.
        
        Args:
            query: Query to retrieve memories for
            max_chars: Maximum characters in context block
            
        Returns:
            Formatted context block string
        """
        if self._drift_detected:
            return ""
            
        memories = self.retrieve(query, limit=5)
        
        if not memories:
            return ""
            
        # Build context block
        lines = ["[MEMORY CONTEXT]"]
        current_len = len(lines[0]) + 2
        
        for mem in memories:
            text = mem.get('text', '')
            kind = mem.get('kind', 'fact')
            scope = mem.get('scope', 'global')
            score = mem.get('score', 0)
            
            # Truncate if needed
            mem_str = f"- [{kind}] {text} (relevance: {score:.2f})"
            
            if current_len + len(mem_str) + 1 > max_chars:
                break
                
            lines.append(mem_str)
            current_len += len(mem_str) + 1
            
        lines.append("[/MEMORY CONTEXT]")
        
        return "\n".join(lines)
        
    def should_write(self, text: str, response: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should write a memory based on message content.
        
        Decision criteria:
        - Must contain stable fact/preference/task
        - Must have justification
        - Session write limit not exceeded
        - Not duplicate in session
        
        Args:
            text: User's message text
            response: Agent's response (optional, for more context)
            
        Returns:
            Tuple of (should_write, justification)
        """
        # Check session limit
        if self._write_count >= MAX_WRITES_PER_SESSION:
            logger.info(f"[AUDIT] Write rejected: session limit exceeded ({self._write_count}/{MAX_WRITES_PER_SESSION})")
            return False, "Session write limit exceeded"
            
        # Check for stable facts (simple heuristic)
        text_lower = text.lower()
        
        # Indicators of storable content
        fact_indicators = [
            'i prefer', 'i like', 'i hate', 'i love', 'i want',
            'my name is', 'i am', 'i work', 'i live',
            'remember that', 'don\'t forget', 'keep in mind',
            'always', 'never', 'usually', 'sometimes',
            'my favorite', 'my hobby', 'i hate when',
            'call me', 'you can call me', 'address is'
        ]
        
        has_fact = any(ind in text_lower for ind in fact_indicators)
        
        # Check for tasks
        task_indicators = [
            'remind me', 'don\'t forget to', 'make sure to',
            'i need to', 'i should', 'must remember to'
        ]
        
        has_task = any(ind in text_lower for ind in task_indicators)
        
        # Length check
        if len(text) < 10:
            return False, "Message too short"
            
        if len(text) > 500:
            text = text[:500]  # Truncate for analysis
            
        # Determine kind and importance
        if has_task:
            kind = 'task'
            importance = 0.8
            justification = "Task/reminder detected"
        elif has_fact:
            kind = 'fact'
            importance = 0.75
            justification = "Fact/preference detected"
        else:
            # Check if response contains useful info to store
            if response and len(response) > 20:
                kind = 'fact'
                importance = 0.7
                justification = "Response contains factual information"
            else:
                return False, "No storable content detected"
                
        # Check for duplicate in session
        text_hash = str(hash(text))
        if text_hash in self._session_hashes:
            return False, "Duplicate content in session"
            
        return True, justification
        
    def write(self, text: str, kind: str = 'fact', scope: str = 'global', 
              importance: float = 0.7, justification: Optional[str] = None) -> bool:
        """
        Write a memory using controlled_write.
        
        This is the ONLY supported write method - bypass attempts
        will be logged and rejected.
        
        Args:
            text: Memory content
            kind: Memory kind (fact, preference, task)
            scope: Memory scope (global, local, session)
            importance: Importance score (0.0-1.0)
            justification: Why this should be stored
            
        Returns:
            True if write succeeded
        """
        # Log write attempt
        logger.info(f"[AUDIT] Write attempt: session={self.session_id}, kind={kind}, importance={importance}")
        logger.info(f"[AUDIT] Justification: {justification}")
        
        # Check drift
        if self._drift_detected:
            logger.warning(f"[DRIFT] Write blocked - memory disabled")
            return False
            
        # Check policy via policy_layer
        item = {'text': text, 'kind': kind, 'scope': scope, 'importance': importance}
        should, reason = should_persist(item)
        if not should:
            logger.info(f"[AUDIT] Write rejected by policy: {reason}")
            return False
            
        # Check session limit again
        if self._write_count >= MAX_WRITES_PER_SESSION:
            logger.info(f"[AUDIT] Write rejected: session limit")
            return False
            
        # Use controller if available
        if self._controller:
            try:
                success = self._controller.controlled_write(
                    text=text,
                    kind=kind,
                    scope=scope,
                    importance=importance,
                    justification=justification
                )
                if success:
                    self._write_count += 1
                    self._session_hashes.add(str(hash(text)))
                    logger.info(f"[AUDIT] Write success via controller")
                return success
            except Exception as e:
                logger.error(f"Controller write failed: {e}")
                return False
                
        # Fallback to direct client
        try:
            item['timestamp'] = datetime.now(timezone.utc).isoformat()
            result = memory_upsert([item])
            success = result.get('inserted', 0) >= 1
            if success:
                self._write_count += 1
                self._session_hashes.add(str(hash(text)))
                logger.info(f"[AUDIT] Write success via direct client")
            return success
        except Exception as e:
            logger.error(f"Direct write failed: {e}")
            return False
            
    def extract_and_promote_traits(self, user_message: str, assistant_message: str) -> bool:
        """
        Extract user traits and promote to long-term memory.
        
        Only promotes non-sensitive features with high confidence.
        
        Args:
            user_message: User's message
            assistant_message: Assistant's response
            
        Returns:
            True if trait was promoted
        """
        if not USER_MODEL_AVAILABLE or not self._user_model:
            return False
            
        try:
            # Create conversation turn
            turn = ConversationTurn(speaker="user", text=user_message)
            
            # Extract features
            features = extract_features([turn])
            
            # Get strongest features
            strongest = get_strongest_features(features, threshold=0.5)
            
            if not strongest:
                return False
                
            # Update hypotheses
            engine = self._user_model['engine']
            for trait, value in strongest.items():
                engine.observe(trait, value)
                
            # Try to promote high-confidence traits
            promoter = self._user_model['promoter']
            promoted = promoter.promote_if_ready()
            
            if promoted:
                logger.info(f"[USER_MODEL] Promoted traits: {promoted}")
                
            return len(promoted) > 0
            
        except Exception as e:
            logger.error(f"Trait extraction/promotion failed: {e}")
            return False
            
    def get_mode(self) -> str:
        """
        Get current operational mode from mode router.
        
        Returns:
            Mode string (default: 'interactive')
        """
        if not MODE_ROUTER_AVAILABLE or not self._mode_router:
            return 'interactive'
            
        try:
            return self._mode_router.get_current_mode()
        except Exception:
            return 'interactive'
            
    def full_cycle(self, user_message: str, assistant_response: str = "") -> Dict[str, Any]:
        """
        Execute full memory cycle: retrieve -> build context -> decide write.
        
        This is the main entry point for the middleware.
        
        Args:
            user_message: User's incoming message
            assistant_response: Assistant's response (for write decision)
            
        Returns:
            Dict with:
                - context_block: Memory context for prompt injection
                - should_write: Whether to write
                - write_decision: Details of write decision
                - mode: Current operational mode
                - drift_detected: Whether memory is disabled
        """
        logger.info(f"[FULL_CYCLE] Session: {self.session_id}")
        logger.info(f"[FULL_CYCLE] User message: {user_message[:100]}...")
        
        # Check drift
        drift = self.check_drift()
        
        # Get mode
        mode = self.get_mode()
        
        # Build context block (only in interactive mode)
        context_block = ""
        if not drift and mode == 'interactive':
            context_block = self.build_context_block(user_message)
            
        # Decide whether to write
        should_write, justification = self.should_write(user_message, assistant_response)
        
        # Execute write if decided
        write_success = False
        if should_write and assistant_response:
            # Determine kind and importance
            text_lower = user_message.lower()
            if 'remind' in text_lower or 'don\'t forget' in text_lower:
                kind, importance = 'task', 0.8
            elif 'prefer' in text_lower or 'like' in text_lower or 'hate' in text_lower:
                kind, importance = 'preference', 0.75
            else:
                kind, importance = 'fact', 0.7
                
            write_success = self.write(
                text=user_message,
                kind=kind,
                importance=importance,
                justification=justification
            )
            
        # Extract and promote traits (async, doesn't block)
        trait_promoted = False
        if assistant_response:
            try:
                trait_promoted = self.extract_and_promote_traits(user_message, assistant_response)
            except Exception as e:
                logger.error(f"Trait promotion error: {e}")
                
        result = {
            'session_id': self.session_id,
            'drift_detected': drift,
            'mode': mode,
            'context_block': context_block,
            'should_write': should_write,
            'write_decision': {
                'justification': justification,
                'success': write_success,
                'session_write_count': self._write_count,
                'session_limit': MAX_WRITES_PER_SESSION
            },
            'trait_promoted': trait_promoted
        }
        
        logger.info(f"[FULL_CYCLE] Result: drift={drift}, write={write_success}, traits={trait_promoted}")
        
        return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="OpenClaw Memory Middleware")
    parser.add_argument('--action', required=True, 
                        choices=['retrieve', 'write', 'full_cycle', 'health', 'drift_check'],
                        help='Action to perform')
    parser.add_argument('--query', help='Query for retrieval')
    parser.add_argument('--text', help='Text for write')
    parser.add_argument('--kind', default='fact', choices=['fact', 'preference', 'task'],
                        help='Memory kind')
    parser.add_argument('--scope', default='global', choices=['global', 'local', 'session'],
                        help='Memory scope')
    parser.add_argument('--importance', type=float, default=0.7,
                        help='Importance score (0.0-1.0)')
    parser.add_argument('--message', help='User message for full cycle')
    parser.add_argument('--response', help='Assistant response for full cycle')
    parser.add_argument('--session-id', help='Session ID')
    parser.add_argument('--limit', type=int, default=5, help='Limit for retrieval')
    parser.add_argument('--max-chars', type=int, default=1500, help='Max chars for context block')
    
    args = parser.parse_args()
    
    # Initialize middleware
    middleware = MemoryMiddleware(session_id=args.session_id)
    
    if args.action == 'health':
        health = memory_health() if MEMORY_CONTROLLER_AVAILABLE else {'status': 'unavailable'}
        print(json.dumps(health, indent=2))
        
    elif args.action == 'drift_check':
        drift = middleware.check_drift()
        print(json.dumps({'drift_detected': drift}, indent=2))
        
    elif args.action == 'retrieve':
        if not args.query:
            print("Error: --query required for retrieve")
            sys.exit(1)
        results = middleware.retrieve(args.query, args.limit)
        print(json.dumps(results, indent=2))
        
    elif args.action == 'write':
        if not args.text:
            print("Error: --text required for write")
            sys.exit(1)
        success = middleware.write(args.text, args.kind, args.scope, args.importance)
        print(json.dumps({'success': success}, indent=2))
        
    elif args.action == 'full_cycle':
        if not args.message:
            print("Error: --message required for full_cycle")
            sys.exit(1)
        result = middleware.full_cycle(args.message, args.response or "")
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
