"""The committed `ui/dist` bundle, and the check that stops it going stale.

`ui/dist/` is committed so a clean clone on a laptop with the wifi off can boot the demo without
`npm ci`. The price is a build artifact that can drift from its sources without anyone
noticing: someone edits `ui/src`, does not rebuild, and the stage shows last week's UI. Nobody
reads diffs on this project, so the gate has to notice for them.

`scripts/ui_dist.py` does it by content, not by timestamp -- a git checkout gives every file a
fresh mtime, so mtimes say nothing in exactly the clean-clone case that matters. The first test
here is the guardrail on the real tree; the rest hold the checker to what it claims, on a tree
small enough to break one thing at a time.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("ui_dist", ROOT / "scripts" / "ui_dist.py")
ui_dist = importlib.util.module_from_spec(_spec)
sys.modules["ui_dist"] = ui_dist
_spec.loader.exec_module(ui_dist)


def test_the_committed_ui_dist_matches_ui_src() -> None:
    """The guardrail. Red here means: rebuild the bundle and commit it -- `make ui`."""
    found = ui_dist.problems(ROOT)
    assert not found, ("ui/dist cannot be trusted, and a clean clone will serve it as-is:\n  - "
                       + "\n  - ".join(found) + "\nFix: make ui, then commit ui/dist")


def test_the_cli_exits_zero_on_a_current_bundle_and_two_on_bad_usage() -> None:
    """`clean_clone_test.sh` and `make demo` act on this exit status. The failing case (1) is
    covered below through `check()`, which the CLI returns unchanged."""
    ok = subprocess.run([sys.executable, str(ROOT / "scripts/ui_dist.py"), "check"],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert subprocess.run([sys.executable, str(ROOT / "scripts/ui_dist.py"), "nonsense"],
                          capture_output=True, text=True).returncode == 2


# --------------------------------------------------------------------------- #
# The checker, on a tree small enough to break one thing at a time
# --------------------------------------------------------------------------- #

def _tree(root: Path) -> Path:
    """A miniature repo: a UI whose source imports a file from outside `ui/`, and its build."""
    files = {
        "ui/src/main.tsx": 'import geo from "../../data/reference/map.geojson?url";\n',
        "ui/src/App.tsx": "export default function App() { return null }\n",
        "ui/public/fixtures/a.json": "{}",
        "ui/index.html": '<div id="root"></div><script type="module" src="/src/main.tsx"></script>',
        "ui/package.json": "{}", "ui/package-lock.json": "{}", "ui/vite.config.ts": "",
        "ui/tsconfig.json": "{}",
        "data/reference/map.geojson": '{"type": "FeatureCollection"}',
        "ui/dist/index.html": '<div id="root"></div><script src="/assets/app.js"></script>'
                              '<link rel="stylesheet" href="/assets/app.css">',
        "ui/dist/assets/app.js": "1", "ui/dist/assets/app.css": "1",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    assert ui_dist.stamp(root) == 0
    return root


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return _tree(tmp_path)


def test_a_fresh_stamped_build_passes(tree: Path) -> None:
    assert ui_dist.problems(tree) == []
    assert ui_dist.check(tree) == 0


def test_the_stamp_does_not_depend_on_file_times(tree: Path) -> None:
    """Touch every source: a clone rewrites all mtimes, and the answer must not move."""
    for p in (tree / "ui").rglob("*"):
        if p.is_file() and "dist" not in p.parts:
            p.write_bytes(p.read_bytes())
    assert ui_dist.problems(tree) == []


def test_an_edited_source_file_is_named(tree: Path) -> None:
    (tree / "ui/src/App.tsx").write_text("export default function App() { return 1 }\n")
    (found,) = ui_dist.problems(tree)
    assert "changed" in found and "ui/src/App.tsx" in found
    assert ui_dist.check(tree) == 1


def test_a_new_or_deleted_source_file_is_stale(tree: Path) -> None:
    (tree / "ui/src/New.tsx").write_text("export {}\n")
    assert any("added" in p and "ui/src/New.tsx" in p for p in ui_dist.problems(tree))
    (tree / "ui/src/New.tsx").unlink()
    (tree / "ui/src/App.tsx").unlink()
    assert any("removed" in p and "ui/src/App.tsx" in p for p in ui_dist.problems(tree))


def test_the_public_fixtures_the_bundle_copies_are_inputs_too(tree: Path) -> None:
    (tree / "ui/public/fixtures/a.json").write_text('{"changed": true}')
    assert any("ui/public/fixtures/a.json" in p for p in ui_dist.problems(tree))


def test_a_file_outside_ui_that_the_source_imports_is_an_input(tree: Path) -> None:
    """Map.tsx bundles data/reference/nyc_modzcta.geojson; regenerating it makes dist stale."""
    (tree / "data/reference/map.geojson").write_text('{"type": "changed"}')
    assert any("data/reference/map.geojson" in p for p in ui_dist.problems(tree))


def test_the_checker_finds_that_outside_import_rather_than_being_told_about_it(
        tree: Path) -> None:
    assert "data/reference/map.geojson" in ui_dist.inputs(tree)
    (tree / "ui/src/main.tsx").write_text("export {}\n")
    assert "data/reference/map.geojson" not in ui_dist.inputs(tree)


def test_finder_droppings_are_not_sources(tree: Path) -> None:
    (tree / "ui/src/.DS_Store").write_bytes(b"\x00")
    assert ui_dist.problems(tree) == []


def test_a_missing_bundle_is_reported_as_missing(tree: Path) -> None:
    (tree / "ui/dist/index.html").unlink()
    (found,) = ui_dist.problems(tree)
    assert "missing" in found and "no built UI" in found
    assert ui_dist.check(tree) == 1


def test_a_page_that_points_at_a_missing_asset_is_a_broken_bundle(tree: Path) -> None:
    (tree / "ui/dist/assets/app.js").unlink()
    assert any("/assets/app.js" in p and "not in ui/dist" in p for p in ui_dist.problems(tree))


def test_a_build_nobody_stamped_is_not_trusted(tree: Path) -> None:
    """`npm run build` by hand empties dist and drops the manifest with it."""
    (tree / "ui/dist" / ui_dist.MANIFEST).unlink()
    assert any(ui_dist.MANIFEST in p for p in ui_dist.problems(tree))


def test_stamping_refuses_when_there_is_nothing_to_stamp(tmp_path: Path) -> None:
    (tmp_path / "ui").mkdir()
    assert ui_dist.stamp(tmp_path) == 1


def test_the_manifest_is_reproducible(tree: Path) -> None:
    """It is committed, so stamping twice must not produce a diff."""
    first = (tree / "ui/dist" / ui_dist.MANIFEST).read_text()
    ui_dist.stamp(tree)
    assert (tree / "ui/dist" / ui_dist.MANIFEST).read_text() == first
    assert not re.search(r"\d{4}-\d\d-\d\d|/Users/|/tmp/", first), "the manifest carries a stamp"
