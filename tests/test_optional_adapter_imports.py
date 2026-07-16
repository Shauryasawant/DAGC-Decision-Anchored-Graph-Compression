import builtins

import pytest

from dagc.adapters import SentenceTransformerEmbedder, TiktokenTokenizer


def test_tiktoken_error_message_points_to_project_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("No module named 'tiktoken'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install -e '\.\[tiktoken\]'"):
        TiktokenTokenizer("cl100k_base")


def test_sentence_transformers_error_message_points_to_project_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install -e '\.\[sentence-transformers\]'"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")
