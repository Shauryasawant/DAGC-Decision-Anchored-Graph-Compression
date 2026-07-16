# Changelog

All notable changes to this project are documented in this file.

## Unreleased

- Removed commitment extraction, protection, evaluation metrics, and CLI support.

## 0.2.0 - 2026-07-15

- Fix: `src/dagc/__init__.py.__version__` was stale at `0.1.0` while `pyproject.toml` already said `0.1.1`; both now read `0.2.0`.
