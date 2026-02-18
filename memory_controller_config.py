#!/usr/bin/env python3
"""
Memory Controller Configuration

Centralized configuration for memory orchestration layer.
All constants used by MemoryController are defined here.
"""

# Session Limits
MAX_WRITES_PER_SESSION = 5
SESSION_TIMEOUT_MINUTES = 30

# Feature Toggles
RETRIEVAL_ENABLED = True
WRITE_ENABLED = True

# Policy Thresholds (should match policy_layer)
MIN_IMPORTANCE = 0.7
SIMILARITY_THRESHOLD = 0.30

# Context Building
MAX_CONTEXT_ITEMS = 8
MAX_CONTEXT_CHARS = 1200

# Audit Logging
AUDIT_LOG_PATH = "logs/memory_audit.log"
AUDIT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
AUDIT_LOG_BACKUP_COUNT = 5

# Drift Detection
REQUIRED_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
REQUIRED_DIMENSION = 768

# Retrieval Decision Patterns
# These determine when retrieval should be allowed
RETRIEVAL_TRIGGERS = [
    # Identity continuity
    "remember",
    "preference",
    "like",
    "prefer",
    "usually",
    "always",
    "never",
    
    # Project context
    "project",
    "work",
    "codebase",
    "repository",
    "setup",
    "configure",
    
    # Long-term knowledge
    "fact",
    "know",
    "learned",
    "discovered",
    "found",
    "verified",
    
    # Task continuity
    "task",
    "todo",
    "goal",
    "objective",
    "complete",
    "finish",
    "resume",
    
    # Context questions
    "what did",
    "when did",
    "how did",
    "did we",
    "have we",
    "previous",
    "last time",
    "before",
]

# These patterns suggest retrieval should NOT happen
RETRIEVAL_EXCLUSIONS = [
    # Casual chat
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "please",
    "ok",
    "okay",
    "goodbye",
    "bye",
    
    # Ephemeral
    "now",
    "current",
    "today",
    "right now",
    "at the moment",
    
    # Tool-only
    "run",
    "execute",
    "show me",
    "display",
    "list",
    "get",
    "fetch",
]

# Write justification requirements
REQUIRE_WRITE_JUSTIFICATION = True
MIN_JUSTIFICATION_LENGTH = 10

# Duplicate detection
DUPLICATE_WINDOW_HOURS = 1  # Consider duplicates within this window

# Session management
SESSION_ID_LENGTH = 16
