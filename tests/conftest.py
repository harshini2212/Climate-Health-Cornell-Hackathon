"""Session setup for the suite.

The one job here is making `tests/tables.py` importable from every test module regardless
of pytest's import mode, so no test has to reach into `data/` to find a frame to work with.
See that module for why the suite builds its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
