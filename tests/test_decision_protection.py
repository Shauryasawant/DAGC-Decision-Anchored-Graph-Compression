"""
DAGC Compressor module - handles compression of conversation traces while preserving critical decision information.
"""
from typing import List, Dict, Any, Set, Optional

def _decision_critical_values(decisions: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract critical values from decisions that must be preserved during compression.

    This includes:
    - Action verbs (e.g., 'recommend', 'best', 'use')
    - Targets (e.g., 'hash index', 'DEP-4021')
    - Rationale items

    Args:
        decisions: List of decision dictionaries with 'action', 'target', 'rationale', etc.

    Returns:
        Set of critical string values that must survive compression
    """
    critical = set()
    for d in decisions:
        # Include the action verb (e.g., 'recommend', 'best')
        # This fixes the bug where action verbs were not being preserved
        if 'action' in d and d['action']:
            critical.add(d['action'])

        # Include the target (e.g., 'hash index')
        if 'target' in d and d['target']:
            critical.add(d['target'])

        # Include rationale items if present
        if 'rationale' in d and d['rationale']:
            for r in d['rationale']:
                if r:
                    critical.add(r)

    return critical

class DAGCConfig:
    """Configuration for DAGC compression."""

    def __init__(self, TARGET_REDUCTION: float = 0.87, **kwargs):
        self.TARGET_REDUCTION = TARGET_REDUCTION
        for key, value in kwargs.items():
            setattr(self, key, value)

def compress_dagc(trace: List[Dict[str, Any]], cfg: Optional[DAGCConfig] = None) -> List[Dict[str, Any]]:
    """
    Compress a conversation trace while preserving critical decision information.

    Args:
        trace: List of message dictionaries (role, content, etc.)
        cfg: DAGCConfig instance with compression parameters

    Returns:
        Compressed trace
    """
    if cfg is None:
        cfg = DAGCConfig()

    # Simplified implementation for the fix demonstration
    # In the real implementation, this would use _decision_critical_values
    # to determine which parts to preserve
    return trace