import os

from dagc import wrap


def test_openai_upstream_accepts_a_v1_suffix():
    config = wrap._TOOLS["codex"]
    assert wrap._normalize_upstream("https://api.openai.com/v1/", config) == "https://api.openai.com"
    assert wrap._normalize_upstream("https://gateway.example", config) == "https://gateway.example"


def test_claude_does_not_strip_a_path_from_its_upstream():
    config = wrap._TOOLS["claude"]
    assert wrap._normalize_upstream("https://gateway.example/v1", config) == "https://gateway.example/v1"


def test_wrap_codex_sets_only_codex_base_url(monkeypatch):
    captured = {}

    class FakeProcess:
        def poll(self):
            return 0

    class FakeRunResult:
        returncode = 42

    monkeypatch.setattr(wrap.shutil, "which", lambda executable: "/usr/bin/codex")
    monkeypatch.setattr(wrap.subprocess, "Popen", lambda *args, **kwargs: captured.setdefault("proxy", (args, kwargs)) and FakeProcess())
    monkeypatch.setattr(wrap, "_wait_for_proxy", lambda process, port: None)
    monkeypatch.setattr(wrap, "_stop_proxy", lambda process: captured.setdefault("stopped", process))

    def fake_run(command, *, env, check):
        captured["command"] = command
        captured["environment"] = env
        captured["check"] = check
        return FakeRunResult()

    monkeypatch.setattr(wrap.subprocess, "run", fake_run)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unrelated.example")

    assert wrap.wrap_codex(["--version"], port=8765, upstream="https://api.openai.com/v1") == 42
    assert captured["command"] == ["codex", "--version"]
    assert captured["environment"]["OPENAI_BASE_URL"] == "http://127.0.0.1:8765/v1"
    assert captured["environment"]["ANTHROPIC_BASE_URL"] == "https://unrelated.example"
    assert captured["proxy"][1]["env"]["UPSTREAM_BASE_URL"] == "https://api.openai.com"
    assert captured["check"] is False


def test_missing_tool_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(wrap.shutil, "which", lambda executable: None)
    assert wrap.wrap_aider([]) == 127
    assert "Cannot find 'aider' on PATH." in capsys.readouterr().err
