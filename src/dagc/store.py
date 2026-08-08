"""
dagc.store — persistence for reversible retrieval.

compress()/compress_any() tag every surviving message with `_orig_idx`,
its position in the original trace -- but nothing in the compressor
itself keeps the original trace around afterward. That's fine as long
as the caller still has `messages` in memory (dagc.formats.denormalize_trace
already handles that case). It stops being fine the moment compression
happens in one process/request and the "give me back the original text of
message N" need shows up somewhere else -- e.g. a proxy server that
compressed a request an hour ago, and now something (a human reviewing a
log, a different agent) wants to see what message #14 actually said.

MessageStore closes that gap: a minimal, pluggable key-value store keyed
by (trace_id, orig_idx). Dict-backed by default (in-process only, zero
dependency); swap in anything matching the Backend protocol (Redis, a
SQLite table, etc.) for real cross-process persistence.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    def put(self, trace_id: str, orig_idx: int, message: Dict) -> None: ...
    def get(self, trace_id: str, orig_idx: int) -> Optional[Dict]: ...
    def get_trace(self, trace_id: str) -> Dict[int, Dict]: ...


class DictBackend:
    """In-process default. Lost on restart -- fine for a single long-lived
    server process, not for anything that needs to survive a restart or
    be shared across processes. Swap in FileBackend or your own for that."""

    def __init__(self):
        self._data: Dict[str, Dict[int, Dict]] = {}

    def put(self, trace_id: str, orig_idx: int, message: Dict) -> None:
        self._data.setdefault(trace_id, {})[orig_idx] = message

    def get(self, trace_id: str, orig_idx: int) -> Optional[Dict]:
        return self._data.get(trace_id, {}).get(orig_idx)

    def get_trace(self, trace_id: str) -> Dict[int, Dict]:
        return dict(self._data.get(trace_id, {}))


class FileBackend:
    """One JSON file per trace_id under `directory`. Simple, durable,
    no extra dependency -- a reasonable default for a single-machine
    dagc-server deployment. Not safe for concurrent writers across
    processes (last-write-wins, no locking)."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, trace_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in trace_id)
        return os.path.join(self.directory, f"{safe}.json")

    def put(self, trace_id: str, orig_idx: int, message: Dict) -> None:
        path = self._path(trace_id)
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        data[str(orig_idx)] = message
        with open(path, "w") as f:
            json.dump(data, f)

    def get(self, trace_id: str, orig_idx: int) -> Optional[Dict]:
        path = self._path(trace_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get(str(orig_idx))

    def get_trace(self, trace_id: str) -> Dict[int, Dict]:
        path = self._path(trace_id)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}


class MessageStore:
    """Public entry point. Wraps a Backend (DictBackend by default)."""

    def __init__(self, backend: Optional[Backend] = None):
        self.backend = backend or DictBackend()

    def save_trace(self, trace_id: str, original_messages: List[Dict]) -> None:
        """Persist every message in `original_messages`, keyed by its own
        list position -- the same index compress() writes into `_orig_idx`."""
        for idx, msg in enumerate(original_messages):
            self.backend.put(trace_id, idx, msg)

    def get_message(self, trace_id: str, orig_idx: int) -> Optional[Dict]:
        return self.backend.get(trace_id, orig_idx)

    def get_original_trace(self, trace_id: str) -> Dict[int, Dict]:
        return self.backend.get_trace(trace_id)

    def resolve(self, trace_id: str, compressed_messages: List[Dict]) -> List[Dict]:
        """Given compressed output (each carrying `_orig_idx`), return the
        original, uncompressed message for each -- best-effort: falls back
        to the compressed message itself if the store has nothing for that
        index (trace never saved, or since expired)."""
        out = []
        for m in compressed_messages:
            idx = m.get("_orig_idx")
            original = self.backend.get(trace_id, idx) if idx is not None else None
            out.append(original if original is not None else m)
        return out


__all__ = ["MessageStore", "Backend", "DictBackend", "FileBackend"]