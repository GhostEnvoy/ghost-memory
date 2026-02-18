#!/usr/bin/env python3
"""
Mode Router - Assistant/Advisor/Mirror/Twin Behavior Selection

Routes to different behavior modes based on explicit commands or context.
Never defaults to mirror/twin - must be explicitly invoked.
"""

import os
import sys
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_controller import MemoryController
from user_model.features import UserFeatures


# Setup logging
logger = logging.getLogger("mode_router")

# Mode constants
MODE_ASSISTANT = "assistant"
MODE_ADVISOR = "advisor"
MODE_MIRROR = "mirror"
MODE_TWIN = "twin"

VALID_MODES = [MODE_ASSISTANT, MODE_ADVISOR, MODE_MIRROR, MODE_TWIN]

# Rate limiting for mirror mode
MIRROR_RATE_LIMIT_HOURS = 24
MIRROR_MIN_CONFIDENCE = 0.85
MIRROR_MIN_EVIDENCE = 5
MIRROR_MAX_AGE_DAYS = 30

# Advisor auto-trigger keywords (conservative)
ADVISOR_KEYWORDS = [
    "plan", "strategy", "approach", "design", "architecture",
    "decision", "choose", "select", "recommend", "advice",
    "best practice", "pattern", "methodology", "framework"
]


@dataclass
class MirrorUsage:
    """Track mirror mode usage for rate limiting"""
    last_used: str
    count: int = 0


class ModeRouter:
    """
    Routes to appropriate assistant behavior mode.
    
    Modes:
    - assistant: Default, task-focused, retrieves tasks/preferences/facts
    - advisor: Planning/strategy mode, also retrieves high-confidence traits
    - mirror: Reflects user's patterns (rate-limited, high-confidence only)
    - twin: Imitates user's style (explicit invocation only)
    """
    
    def __init__(self, memory_controller: Optional[MemoryController] = None):
        self.mc = memory_controller or MemoryController()
        self.current_mode = MODE_ASSISTANT
        self._mirror_usage_file = "user_model/mirror_usage.json"
        self._mirror_usage = self._load_mirror_usage()
    
    def _load_mirror_usage(self) -> MirrorUsage:
        """Load mirror usage tracking from file"""
        try:
            if os.path.exists(self._mirror_usage_file):
                with open(self._mirror_usage_file, 'r') as f:
                    data = json.load(f)
                    return MirrorUsage(**data)
        except (json.JSONDecodeError, TypeError):
            pass
        
        return MirrorUsage(
            last_used=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            count=0
        )
    
    def _save_mirror_usage(self):
        """Save mirror usage tracking to file"""
        os.makedirs(os.path.dirname(self._mirror_usage_file), exist_ok=True)
        with open(self._mirror_usage_file, 'w') as f:
            json.dump({
                'last_used': self._mirror_usage.last_used,
                'count': self._mirror_usage.count
            }, f)
    
    def _is_mirror_rate_limited(self) -> bool:
        """Check if mirror mode is rate limited"""
        last_used = datetime.fromisoformat(self._mirror_usage.last_used)
        time_since = datetime.now(timezone.utc) - last_used
        return time_since < timedelta(hours=MIRROR_RATE_LIMIT_HOURS)
    
    def _record_mirror_usage(self):
        """Record mirror mode usage"""
        self._mirror_usage.last_used = datetime.now(timezone.utc).isoformat()
        self._mirror_usage.count += 1
        self._save_mirror_usage()
        
        # Log audit
        logger.info(f"Mirror mode activated (total uses: {self._mirror_usage.count})")
    
    def detect_mode(self, user_input: str, features: Optional[UserFeatures] = None) -> str:
        """
        Detect appropriate mode from user input.
        
        Explicit commands override everything:
        - /assistant, /advisor, /mirror, /twin
        
        Args:
            user_input: The user's message
            features: Optional extracted features for context
            
        Returns:
            Mode string (assistant/advisor/mirror/twin)
        """
        input_lower = user_input.lower().strip()
        
        # Check for explicit mode commands
        if input_lower.startswith("/assistant"):
            return MODE_ASSISTANT
        
        if input_lower.startswith("/advisor"):
            return MODE_ADVISOR
        
        if input_lower.startswith("/mirror"):
            return MODE_MIRROR
        
        if input_lower.startswith("/twin"):
            return MODE_TWIN
        
        # No explicit command - use context
        # Default to assistant
        detected_mode = MODE_ASSISTANT
        
        # Check for advisor keywords (only if in work mode)
        if features and features.work_mode >= 0.6:
            for keyword in ADVISOR_KEYWORDS:
                if keyword in input_lower:
                    detected_mode = MODE_ADVISOR
                    break
        
        return detected_mode
    
    def set_mode(self, mode: str) -> bool:
        """
        Set mode explicitly.
        
        Args:
            mode: Mode to set
            
        Returns:
            True if mode was set successfully
        """
        if mode not in VALID_MODES:
            logger.warning(f"Invalid mode: {mode}")
            return False
        
        # Check rate limiting for mirror
        if mode == MODE_MIRROR and self._is_mirror_rate_limited():
            logger.warning("Mirror mode rate limited")
            return False
        
        self.current_mode = mode
        
        # Record usage for mirror
        if mode == MODE_MIRROR:
            self._record_mirror_usage()
        
        # Log audit for mirror/twin
        if mode in [MODE_MIRROR, MODE_TWIN]:
            logger.info(f"Mode changed to {mode}")
        
        return True
    
    def get_retrieval_params(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get retrieval parameters for a mode.
        
        Args:
            mode: Mode to get params for (defaults to current_mode)
            
        Returns:
            Dict with retrieval configuration
        """
        mode = mode or self.current_mode
        
        base_params = {
            'kinds': ['fact', 'preference', 'task'],
            'scopes': ['global', 'local', 'session'],
            'min_importance': 0.7,
        }
        
        if mode == MODE_ASSISTANT:
            # Standard retrieval
            return {
                **base_params,
                'include_traits': False,
                'limit': 5,
            }
        
        elif mode == MODE_ADVISOR:
            # Include high-confidence traits
            return {
                **base_params,
                'include_traits': True,
                'trait_min_confidence': 0.75,
                'trait_min_evidence': 3,
                'limit': 8,
            }
        
        elif mode == MODE_MIRROR:
            # Only high-confidence, recent traits
            return {
                **base_params,
                'include_traits': True,
                'trait_min_confidence': MIRROR_MIN_CONFIDENCE,
                'trait_min_evidence': MIRROR_MIN_EVIDENCE,
                'trait_max_age_days': MIRROR_MAX_AGE_DAYS,
                'kinds': ['fact'],  # Only traits encoded as facts
                'scopes': ['global'],
                'limit': 5,
            }
        
        elif mode == MODE_TWIN:
            # Load style profile (derived from traits)
            return {
                **base_params,
                'include_traits': True,
                'load_style_profile': True,
                'limit': 3,
            }
        
        return base_params
    
    def get_mode_context(self, mode: Optional[str] = None) -> str:
        """
        Get context string for a mode.
        
        Args:
            mode: Mode to get context for
            
        Returns:
            Context string describing mode behavior
        """
        mode = mode or self.current_mode
        
        contexts = {
            MODE_ASSISTANT: "You are in assistant mode. Focus on completing tasks efficiently using user's preferences and established facts.",
            MODE_ADVISOR: "You are in advisor mode. Consider user's behavioral patterns and traits when providing strategic recommendations.",
            MODE_MIRROR: "You are in mirror mode. Reflect back the user's established patterns and traits with high confidence. Be insightful but not presumptuous.",
            MODE_TWIN: "You are in twin mode. Adopt the user's communication style and decision-making patterns. Mirror their directness, verbosity, and approach.",
        }
        
        return contexts.get(mode, contexts[MODE_ASSISTANT])
    
    def can_activate_mirror(self) -> tuple[bool, str]:
        """
        Check if mirror mode can be activated.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if self._is_mirror_rate_limited():
            last_used = datetime.fromisoformat(self._mirror_usage.last_used)
            next_available = last_used + timedelta(hours=MIRROR_RATE_LIMIT_HOURS)
            return False, f"Mirror rate limited. Available after {next_available}"
        
        return True, "Mirror available"
    
    def get_current_mode(self) -> str:
        """Get current mode"""
        return self.current_mode
    
    def get_mirror_stats(self) -> Dict[str, Any]:
        """Get mirror mode statistics"""
        last_used = datetime.fromisoformat(self._mirror_usage.last_used)
        time_since = datetime.now(timezone.utc) - last_used
        
        return {
            'total_uses': self._mirror_usage.count,
            'last_used': self._mirror_usage.last_used,
            'hours_since_last_use': time_since.total_seconds() / 3600,
            'rate_limited': self._is_mirror_rate_limited(),
            'hours_until_available': max(0, MIRROR_RATE_LIMIT_HOURS - time_since.total_seconds() / 3600),
        }


if __name__ == "__main__":
    # Self-test
    print("Mode Router Self-Test\n")
    
    router = ModeRouter()
    
    # Test explicit commands
    test_inputs = [
        "/assistant help me code",
        "/advisor how should I architect this",
        "/mirror what do I usually do",
        "/twin respond like I would",
        "Just a regular question",
        "plan this project for me",
    ]
    
    print("Mode Detection:")
    for inp in test_inputs:
        mode = router.detect_mode(inp)
        print(f"  '{inp[:40]}...' → {mode}")
    
    print("\n\nRetrieval Params by Mode:")
    for mode in VALID_MODES:
        params = router.get_retrieval_params(mode)
        print(f"  {mode}: {params}")
    
    print("\n\nMirror Stats:")
    stats = router.get_mirror_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Self-test complete")
