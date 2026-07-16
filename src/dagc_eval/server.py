"""
Minimal wire-compatible proxy: compress the message array inside any
LLM API request body, forward everything else untouched.

Never inspects vendor identity. It locates the message list generically
(via normalize_trace's envelope detection), compresses it, denormalizes
back into the exact original shape, and re-inserts it at the same key --
so this works against any provider whose request body carries a
messages-like array under a recognizable key, not a hardcoded list of
known APIs.

Fail-safe by design: if ANYTHING in the compression path raises, the
original request body is forwarded unmodified rather than erroring --
a proxy sitting in someone's live request path should never 500 due to
a bug in compression logic.
"""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
except ImportError as e:
    raise ImportError(
        "dagc_eval.server needs the 'server' extra. Install it with "
        "pip install 'dagc[server]'"
    ) from e

import dagc
from dagc_eval.normalize import denormalize_trace, normalize_trace

logger = logging.getLogger("dagc_eval.server")

app = FastAPI()

TARGET_REDUCTION = 0.7

_ENVELOPE_KEYS = ('messages', 'trace', 'conversation', 'turns')
_HOP_BY_HOP_HEADERS = {
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailers', 'transfer-encoding', 'upgrade', 'host', 'content-length',
    'cookie', 'set-cookie'
}


def _find_message_key(body: Dict) -> str | None:
    for key in _ENVELOPE_KEYS:
        if isinstance(body.get(key), list):
            return key
    return None


def _select_upstream_headers(headers: Mapping[str, str] | None) -> Dict[str, str]:
    if not headers:
        return {}

    forwarded: Dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in _HOP_BY_HOP_HEADERS:
            continue
        if name.lower() in {'content-type', 'accept', 'authorization', 'x-api-key', 'x-api-token'}:
            forwarded[name] = value
        elif name.lower().startswith('x-'):
            forwarded[name] = value
    return forwarded


def _parse_upstream_routes(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items() if isinstance(v, (str, int, float))}
    return {}


def _resolve_upstream_base_url(path: str, headers: Mapping[str, str] | None = None) -> str:
    header_override = None
    if headers:
        header_override = headers.get('x-dagc-upstream-url') or headers.get('X-DAGC-UPSTREAM-URL')
    if header_override:
        return str(header_override)

    routes = _parse_upstream_routes(os.getenv('DAGC_UPSTREAM_BASE_URLS'))
    if not routes:
        return os.getenv('UPSTREAM_BASE_URL', '')

    for prefix, url in routes.items():
        if path.startswith(f'/{prefix}') or path.startswith(prefix):
            return url
    return os.getenv('UPSTREAM_BASE_URL', '')


def _compress_body(body: Dict) -> Dict:
    """
    Returns a new body dict with its message array compressed, or the
    ORIGINAL body unchanged if anything about this request doesn't look
    compressible (no message array found) or if compression itself
    raises for any reason. Never lets a compression bug break the
    request -- worst case is "forwarded uncompressed", never an error.
    """
    key = _find_message_key(body)
    if key is None:
        return body

    original_messages = body[key]
    try:
        normalized = normalize_trace(original_messages)
        compressed = dagc.compress(normalized, target_reduction=TARGET_REDUCTION)
        denormalized = denormalize_trace(original_messages, compressed)
    except Exception:
        logger.exception("dagc compression failed; forwarding original request uncompressed")
        return body

    new_body = dict(body)
    new_body[key] = denormalized
    return new_body


@app.post("/{path:path}")
async def proxy(path: str, request: Request):
    raw_body = await request.body()
    request_headers = dict(request.headers)
    forwarded_headers = _select_upstream_headers(request_headers)
    upstream_base_url = _resolve_upstream_base_url(f"/{path}", request_headers)

    try:
        body = await request.json()
    except Exception:
        body = None

    if isinstance(body, dict):
        outbound_body = _compress_body(body)
    else:
        outbound_body = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        if outbound_body is not None:
            upstream_resp = await client.post(
                f"{upstream_base_url.rstrip('/')}/{path.lstrip('/')}",
                json=outbound_body,
                headers=forwarded_headers,
            )
        else:
            upstream_resp = await client.post(
                f"{upstream_base_url.rstrip('/')}/{path.lstrip('/')}",
                content=raw_body,
                headers=forwarded_headers,
            )

    if upstream_resp.headers.get('content-type', '').startswith('text/event-stream'):
        return StreamingResponse(
            iter([upstream_resp.content]),
            status_code=upstream_resp.status_code,
            headers=dict(upstream_resp.headers),
        )

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
    )


def run() -> None:
    """
    Entry point for the `dagc-server` console script.

    Configure via environment variables:
      UPSTREAM_BASE_URL         default upstream (e.g. https://api.openai.com)
      DAGC_UPSTREAM_BASE_URLS   JSON map of path-prefix -> upstream base URL,
                                 for routing multiple providers off one proxy
      DAGC_SERVER_HOST          default 0.0.0.0
      DAGC_SERVER_PORT          default 8000
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "dagc-server needs uvicorn. Install it with pip install 'dagc[server]'"
        ) from e

    host = os.getenv("DAGC_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("DAGC_SERVER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()