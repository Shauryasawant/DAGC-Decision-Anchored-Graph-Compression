"""
dagc.rescue_redis — horizontal-scaling counterpart to rescue.py section 8's
in-process `_rescue_sessions` dict.

THE GAP THIS CLOSES: compress(enable_rescue=True) keeps rescue state
(ShadowBuffer + RescueEngine, i.e. the decayed-recurrence tracker and the
capacity-bounded GuaranteedSet) in a plain dict inside the dagc process.
Fine for one process. Once you run more than one dagc-mcp instance (for
throughput or HA), two instances handling the same session_id have two
INDEPENDENT rescue histories -- a decision rescued/promoted on instance A
is invisible to instance B, silently. Sticky routing at the load balancer
avoids this without any code change, but is a routing-layer guarantee, not
a code-level one -- it breaks silently on failover/rebalance. This module
makes rescue state itself shared, so correctness doesn't depend on routing.

WHY PICKLE, AND WHY THAT'S OK HERE, SPECIFICALLY:
ShadowBuffer/RescueEngine/DecayedRecurrenceTracker/GuaranteedSet are live
Python objects, not JSON-shaped data -- serializing them for real means
pickling (or a full rewrite of rescue.py's internals to be JSON-native,
out of scope here). pickle.loads on attacker-controlled bytes is arbitrary
code execution -- that is a real, standard caveat, not boilerplate.
It's acceptable in THIS specific spot because:
  1. The only writer of these Redis keys is this module, from inside your
     own trusted server fleet.
  2. session_id is never itself deserialized -- it's a plain string used
     as a key name.
  3. Nothing here exposes a raw "load this session_id's blob" path to an
     external caller; it's only ever read back by run_rescue_for_call's
     own logic.
If your deployment lets untrusted callers write directly into this Redis
keyspace (shared Redis instance, no auth/ACL separating dagc's keys from
other tenants), do not use this as-is -- put dagc's rescue keys in a
Redis instance/DB/ACL that only this server can write to.

CONCURRENCY: process_turn mutates the tracker/GuaranteedSet/ShadowBuffer
in place -- a classic read-modify-write. Across processes this needs the
same protection FileBackend was missing for trace storage, except here a
lost update means real rescue-state corruption (a decision promoted
twice, two calls stomping the same GuaranteedSet slot), not just a stale
read. session() below holds a Redis distributed lock (SET NX PX -- the
same primitive RedLock is built from; single-instance-Redis version, not
the multi-node RedLock algorithm -- fine for one Redis primary, revisit
if you run Redis Cluster/Sentinel with failover mid-lock) for the entire
compress() call, not just the process_turn step, so a session_id's state
is strictly serialized across the whole fleet.

Requires: pip install redis
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

from .rescue import RescueEngine, RescueEvent, ShadowBuffer, UnrescuableEviction


class RedisRescueSessionStore:
    def __init__(self, url: str, key_prefix: str = "dagc:rescue_session:",
                 ttl_seconds: int = 7 * 24 * 60 * 60,  # 1 week idle expiry
                 lock_timeout_s: float = 15.0):
        import redis  # deferred import -- redis is an optional dependency
        self._redis = redis.from_url(url)
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.lock_timeout_s = lock_timeout_s

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def _load(self, session_id: str, engine_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        raw = self._redis.get(self._key(session_id))
        if raw is not None:
            return pickle.loads(raw)
        return {
            "shadow": ShadowBuffer(),
            "engine": RescueEngine(**(engine_kwargs or {})),
            "last_compressed": [],
            "n_seen": 0,
        }

    def _save(self, session_id: str, sess: Dict[str, Any]) -> None:
        self._redis.set(self._key(session_id), pickle.dumps(sess), ex=self.ttl_seconds)

    def reset(self, session_id: str) -> None:
        self._redis.delete(self._key(session_id))

    def session(self, session_id: str, engine_kwargs: Optional[Dict] = None) -> "_SessionCtx":
        """Context manager: acquires the distributed lock, loads (or
        creates) session state, and saves + releases on clean exit.
        On any exception inside the `with` block, state is NOT saved --
        the next caller retries against the last known-good state rather
        than persisting a possibly-partial mutation."""
        return _SessionCtx(self, session_id, engine_kwargs)


class _SessionCtx:
    def __init__(self, store: RedisRescueSessionStore, session_id: str,
                 engine_kwargs: Optional[Dict]):
        self._store = store
        self.session_id = session_id
        self._engine_kwargs = engine_kwargs
        self._lock = None
        self.sess: Dict[str, Any] = {}

    def __enter__(self) -> "_SessionCtx":
        self._lock = self._store._redis.lock(
            f"{self._store._key(self.session_id)}:lock",
            timeout=self._store.lock_timeout_s)
        acquired = self._lock.acquire(blocking=True, blocking_timeout=self._store.lock_timeout_s)
        if not acquired:
            raise TimeoutError(
                f"could not acquire rescue session lock for {self.session_id!r} "
                f"within {self._store.lock_timeout_s}s -- another process is "
                f"holding it far longer than one compress() call should take. "
                f"Check for a crashed holder or raise lock_timeout_s.")
        self.sess = self._store._load(self.session_id, self._engine_kwargs)
        return self

    def process_new_messages(
        self, messages: List[Dict], budget_tokens: int,
    ) -> Tuple[Set[str], List[RescueEvent], List[UnrescuableEviction]]:
        """Same diff-against-n_seen logic as rescue._run_rescue_for_call,
        operating on this context's loaded session state in place."""
        sess = self.sess
        if len(messages) < sess["n_seen"]:
            # Shorter than what we've seen -- different trace reusing
            # this session_id. Reset rather than diff against the wrong
            # prefix (mirrors the in-process version's same guard).
            sess.update({
                "shadow": ShadowBuffer(),
                "engine": RescueEngine(**(self._engine_kwargs or {})),
                "last_compressed": [],
                "n_seen": 0,
            })

        new_msgs = messages[sess["n_seen"]:]
        force_preserve_total: Set[str] = set()
        events_total: List[RescueEvent] = []
        unrescuable_total: List[UnrescuableEviction] = []

        for msg in new_msgs:
            fp, events, unrescuable = sess["engine"].process_turn(
                new_message=msg, shadow=sess["shadow"],
                last_compressed_messages=sess["last_compressed"],
                compression_budget_tokens=budget_tokens)
            force_preserve_total |= fp
            events_total.extend(events)
            unrescuable_total.extend(unrescuable)

        sess["n_seen"] = len(messages)
        return force_preserve_total, events_total, unrescuable_total

    def set_last_compressed(self, result: List[Dict]) -> None:
        self.sess["last_compressed"] = result

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._store._save(self.session_id, self.sess)
        finally:
            try:
                self._lock.release()
            except Exception:
                pass  # lock may have already expired via its own timeout -- fine