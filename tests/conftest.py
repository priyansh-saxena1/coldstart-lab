"""Shared fixtures.

The default fixtures build a *synthetic* safetensors checkpoint on disk so the
loader/experiment tests run in milliseconds with no network. A separate
``network`` marker gates the one test that pulls a real (tiny) model.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    """A small multi-tensor safetensors file on disk.

    Returns the directory containing ``model.safetensors``.
    """
    import torch
    from safetensors.torch import save_file

    tensors = {
        "layer.0.weight": torch.randn(256, 256, dtype=torch.float32),
        "layer.0.bias": torch.randn(256, dtype=torch.float32),
        "layer.1.weight": torch.randn(256, 256, dtype=torch.float32),
        "embedding.weight": torch.randn(512, 256, dtype=torch.float32),
    }
    d = tmp_path / "ckpt"
    d.mkdir()
    save_file(tensors, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture
def hf_token():
    return os.environ.get("HF_TOKEN")


@pytest.fixture
def url(tmp_path):
    """A SQLite ledger URL for coordinator tests.

    The coordinator's SQL is deliberately portable, so the same tests run here
    against SQLite (fast, no service) and in CI against Postgres by overriding
    COLDSTART_TEST_DB_URL. The Postgres run is the one that exercises the
    tz-aware `now()` path, which SQLite cannot reproduce.
    """
    import os

    override = os.environ.get("COLDSTART_TEST_DB_URL")
    if override:
        from coldstart_lab.distributed.config import normalise_db_url
        from coldstart_lab.distributed.coordinator import _META
        from sqlalchemy import create_engine

        u = normalise_db_url(override)
        eng = create_engine(u)
        _META.drop_all(eng)
        return u
    return f"sqlite:///{tmp_path}/ledger.db"
