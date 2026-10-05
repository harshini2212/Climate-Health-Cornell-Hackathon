"""Parallel test workers must not share a cache directory (see tests/conftest.py)."""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif("PYTEST_XDIST_WORKER" not in os.environ, reason="only meaningful under -n")
def test_each_xdist_worker_has_its_own_cache_dir():
    worker = os.environ["PYTEST_XDIST_WORKER"]
    cache = os.environ.get("XDG_CACHE_HOME", "")
    assert f"leeward-cache-{worker}-" in cache, (
        "xdist workers share a cache dir; ArviZ's daily-warning rename will race between them"
    )
