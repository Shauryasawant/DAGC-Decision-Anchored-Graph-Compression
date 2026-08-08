"""Rescue validation helper for the packaged DAGC distribution.

This module exists so the ``dagc-rescue`` console entry point resolves cleanly
when installed from PyPI. It intentionally stays lightweight: it validates the
rescue API is importable and that a minimal compress/rescue path runs without
raising an exception.
"""
from __future__ import annotations

from dagc import compress
from dagc.rescue import reset_rescue_session


def _smoke_validate() -> None:
    """Run a minimal, in-memory rescue smoke test."""
    messages = [
        {"role": "user", "content": "Please check account 42 and refund the fee."},
        {"role": "assistant", "content": "I will inspect account 42 and confirm the refund."},
        {"role": "assistant", "content": "Account 42 shows a $25 fee; I recommend refunding it."},
    ]
    reset_rescue_session("validate_rescue_smoke")
    result = compress(messages, target_reduction=0.5, session_id="validate_rescue_smoke")
    if not isinstance(result, list):
        raise TypeError("compress() returned a non-list result during rescue smoke validation")


def main() -> int:
    """Entry point for the ``dagc-rescue`` script."""
    _smoke_validate()
    print("Rescue validation helper: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
