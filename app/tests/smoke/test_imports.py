"""Smoke tests for critical imports."""


def test_imports() -> None:
    import pytest

    pytest.importorskip("sentence_transformers")

    import genie_lamp  # noqa: F401
    from genie_lamp.app import create_app
    from genie_lamp.core.vector_store import VectorStore
    from genie_lamp.core.config import load_config

    assert callable(create_app)
    cfg = load_config()
    assert "memory" in cfg
    cfg = {"retrieval": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
    store = VectorStore(cfg)
    assert hasattr(store, "search")
