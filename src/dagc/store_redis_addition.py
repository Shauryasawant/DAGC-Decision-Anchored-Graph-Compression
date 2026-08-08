"""
Addition to dagc/store.py — add this class alongside DictBackend/FileBackend,
and add "RedisBackend" to store.py's __all__.

Requires: pip install redis
"""
from __future__ import annotations

import json
from typing import Dict, Optional


class RedisBackend:
    """Redis-backed Backend for MessageStore -- durable AND shared across
    processes/machines, unlike FileBackend.

    Design choice over FileBackend: uses a Redis HASH per trace_id
    (key f"{key_prefix}{trace_id}", field=str(orig_idx)) instead of a
    single JSON blob. HSET is atomic per-field at the Redis level, so
    concurrent writers touching DIFFERENT orig_idx values in the same
    trace never race each other -- there's no read-modify-write cycle
    to lose. (Two writers touching the SAME orig_idx still resolve
    last-write-wins, same as any KV store; that's expected, not a bug.)

    ttl_seconds: if set, each trace expires after this many seconds of
    no writes (industry-standard practice for ephemeral session/log
    data instead of unbounded growth). None = never expires -- fine for
    low volume, but you likely want a TTL in production.
    """

    def __init__(self, url: str, ttl_seconds: Optional[int] = None,
                 key_prefix: str = "dagc:trace:"):
        import redis  # deferred import -- redis is an optional dependency
        self._redis = redis.from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    def _key(self, trace_id: str) -> str:
        return f"{self.key_prefix}{trace_id}"

    def put(self, trace_id: str, orig_idx: int, message: Dict) -> None:
        key = self._key(trace_id)
        self._redis.hset(key, str(orig_idx), json.dumps(message))
        if self.ttl_seconds:
            self._redis.expire(key, self.ttl_seconds)

    def put_many(self, trace_id: str, messages: Dict[int, Dict]) -> None:
        """Batch write via pipeline -- what MessageStore.save_trace should
        prefer when the backend supports it (see store.py patch note)."""
        key = self._key(trace_id)
        pipe = self._redis.pipeline()
        for idx, msg in messages.items():
            pipe.hset(key, str(idx), json.dumps(msg))
        if self.ttl_seconds:
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def get(self, trace_id: str, orig_idx: int) -> Optional[Dict]:
        val = self._redis.hget(self._key(trace_id), str(orig_idx))
        return json.loads(val) if val is not None else None

    def get_trace(self, trace_id: str) -> Dict[int, Dict]:
        raw = self._redis.hgetall(self._key(trace_id))
        return {int(k): json.loads(v) for k, v in raw.items()}


# --- Patch note for MessageStore.save_trace in store.py --------------------
# Current implementation loops m.put() one at a time. Works fine against
# RedisBackend (each put is still atomic), but for large traces prefer:
#
#     def save_trace(self, trace_id, original_messages):
#         if hasattr(self.backend, "put_many"):
#             self.backend.put_many(trace_id, dict(enumerate(original_messages)))
#         else:
#             for idx, msg in enumerate(original_messages):
#                 self.backend.put(trace_id, idx, msg)
#
# This is backward compatible -- DictBackend/FileBackend don't define
# put_many, so they silently keep using the per-message loop.