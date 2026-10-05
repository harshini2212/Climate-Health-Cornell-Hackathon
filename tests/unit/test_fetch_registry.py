"""Every committed reference table must still have the fetcher that rebuilds it.

A fetcher deleted from `scripts/fetch_sources.py` leaves its parquet and manifest entry in
place, so nothing else notices until someone needs to refresh the data. No network: this
only imports the script and reads the manifest.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_manifest_entry_has_a_registered_fetcher() -> None:
    spec = importlib.util.spec_from_file_location("fetch_sources",
                                                  ROOT / "scripts" / "fetch_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((ROOT / "data" / "reference" / "manifest.json").read_text())
    orphans = sorted(set(manifest) - set(module.REGISTRY))
    assert not orphans, f"manifest entries with no fetcher in scripts/fetch_sources.py: {orphans}"
