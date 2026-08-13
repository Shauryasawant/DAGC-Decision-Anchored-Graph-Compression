"""Launch supported coding CLIs through a local DAGC compression proxy.

The supported tools expose an environment variable that changes the model API
base URL.  ``dagc wrap`` starts :mod:`dagc_eval.server`, points that variable
at the local proxy, and then runs the requested tool unchanged.

Cursor and GitHub Copilot CLI deliberately do not appear here: their model
traffic is sent to provider-controlled backends rather than a user-configured
base URL, so presenting them as proxyable would be misleading.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


_LOCAL_HOST = "127.0.0.1"
_STARTUP_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class _ToolConfig:
    executable: str
    base_url_variable: str
    proxy_path_prefix: str
    default_upstream: str


_TOOLS: Mapping[str, _ToolConfig] = {
    "claude": _ToolConfig(
        executable="claude",
        base_url_variable="ANTHROPIC_BASE_URL",
        proxy_path_prefix="",
        default_upstream="https://api.anthropic.com",
    ),
    "codex": _ToolConfig(
        executable="codex",
        base_url_variable="OPENAI_BASE_URL",
        proxy_path_prefix="/v1",
        default_upstream="https://api.openai.com",
    ),
    "aider": _ToolConfig(
        executable="aider",
        base_url_variable="OPENAI_API_BASE",
        proxy_path_prefix="/v1",
        default_upstream="https://api.openai.com",
    ),
}


def _pick_port() -> int:
    """Ask the OS for an available loopback port.

    The socket is intentionally closed before uvicorn starts; a competing
    process could theoretically claim it in between, in which case startup
    reports a useful error instead of sending a tool to the wrong endpoint.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_LOCAL_HOST, 0))
        return int(sock.getsockname()[1])


def _proxy_base_url(port: int, path_prefix: str) -> str:
    return f"http://{_LOCAL_HOST}:{port}{path_prefix}"


def _normalize_upstream(upstream: str, config: _ToolConfig) -> str:
    """Avoid doubling ``/v1`` when an OpenAI-compatible upstream is supplied.

    Codex and Aider are pointed at ``<proxy>/v1``.  The proxy retains that
    prefix when forwarding, so its upstream needs to be the server root.
    Accepting either ``https://host`` or ``https://host/v1`` is less surprising
    for users accustomed to OpenAI-compatible endpoint configuration.
    """
    result = upstream.rstrip("/")
    if config.proxy_path_prefix and result.endswith(config.proxy_path_prefix):
        return result[: -len(config.proxy_path_prefix)]
    return result


def _wait_for_proxy(process: subprocess.Popen[object], port: int) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "DAGC proxy exited during startup. Install its dependencies with "
                "pip install 'dagc[server]' and try again."
            )
        try:
            with socket.create_connection((_LOCAL_HOST, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"DAGC proxy did not become ready on {_LOCAL_HOST}:{port}.")


def _stop_proxy(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _wrap(tool: str, tool_args: Sequence[str], *, port: int | None, upstream: str | None) -> int:
    config = _TOOLS[tool]
    if shutil.which(config.executable) is None:
        print(f"Cannot find '{config.executable}' on PATH.", file=sys.stderr)
        return 127

    if port is None:
        port = _pick_port()
    if not 1 <= port <= 65535:
        print("--port must be between 1 and 65535.", file=sys.stderr)
        return 2

    selected_upstream = _normalize_upstream(upstream or config.default_upstream, config)
    proxy_env = os.environ.copy()
    proxy_env["UPSTREAM_BASE_URL"] = selected_upstream

    proxy_command = [
        sys.executable, "-m", "uvicorn", "dagc_eval.server:app",
        "--host", _LOCAL_HOST, "--port", str(port),
    ]
    proxy_process = subprocess.Popen(proxy_command, env=proxy_env)
    try:
        _wait_for_proxy(proxy_process, port)
        tool_env = os.environ.copy()
        tool_env[config.base_url_variable] = _proxy_base_url(port, config.proxy_path_prefix)
        return subprocess.run([config.executable, *tool_args], env=tool_env, check=False).returncode
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_proxy(proxy_process)


def wrap_claude(tool_args: Sequence[str], *, port: int | None = None, upstream: str | None = None) -> int:
    """Run Claude Code with ``ANTHROPIC_BASE_URL`` routed through DAGC."""
    return _wrap("claude", tool_args, port=port, upstream=upstream)


def wrap_codex(tool_args: Sequence[str], *, port: int | None = None, upstream: str | None = None) -> int:
    """Run Codex CLI with ``OPENAI_BASE_URL`` routed through DAGC."""
    return _wrap("codex", tool_args, port=port, upstream=upstream)


def wrap_aider(tool_args: Sequence[str], *, port: int | None = None, upstream: str | None = None) -> int:
    """Run Aider with ``OPENAI_API_BASE`` routed through DAGC."""
    return _wrap("aider", tool_args, port=port, upstream=upstream)
