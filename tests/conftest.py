"""Session setup for the suite.

The one job here is making `tests/tables.py` importable from every test module regardless
of pytest's import mode, so no test has to reach into `data/` to find a frame to work with.
See that module for why the suite builds its own.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Each pytest-xdist worker gets its own cache directory, set before anything imports ArviZ.
# ArviZ 0.23 writes a once-a-day warning stamp through a fixed temp file in the user cache
# (`~/.cache/arviz/daily_warning.tmp`, then rename). Parallel workers race on that rename and
# one dies with FileNotFoundError, which turned CI red at random on 2026-10-05.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _WORKER:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix=f"leeward-cache-{_WORKER}-")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
