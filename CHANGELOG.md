# Changelog

All notable changes to this project are documented in this file.

## Unreleased

- Packaging fix: include runtime JSON/JSONL assets in the distribution so installed builds work outside the repo checkout.

## 0.1.9 - 2026-08-14

- Preserve short numeric configuration values when their keys or nearby text
  identify them as meaningful settings.
- Retain additional compact tool-call arguments within a character budget.
- Add optional preprocessing helpers for repeated tool calls and large JSON
  tool-result arrays via `dagc.dagc_boost`.

## 0.1.0 - 2026-07-26

- Initial package release for DAGC with compression, rationale extraction, and evaluation support.
