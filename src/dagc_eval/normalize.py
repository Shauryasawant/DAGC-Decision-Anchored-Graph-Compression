"""Backward-compatible normalization helpers for dagc_eval.

The shared implementation now lives in dagc.formats so the core package and
this compatibility module use the same logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dagc.formats import (  # noqa: F401
    normalize_message,
    normalize_trace,
    denormalize_message,
    denormalize_trace,
    register_adapter,
    REGISTRY as ADAPTER_REGISTRY,
)

__all__ = [
    "normalize_message", "normalize_trace",
    "denormalize_message", "denormalize_trace",
    "register_adapter", "ADAPTER_REGISTRY",
]
