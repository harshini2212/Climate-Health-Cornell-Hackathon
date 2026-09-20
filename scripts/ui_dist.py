#!/usr/bin/env python3
"""The committed ui/dist bundle, and whether it still matches the sources it was built from.

    python scripts/ui_dist.py stamp    # after `npm run build`: record what dist was built from
    python scripts/ui_dist.py check    # exit 1, naming the reason, if dist cannot be trusted

`ui/dist/` is committed so a clean clone with the wifi off can `make demo` without `npm ci`.
A committed build artifact rots silently: someone edits ui/src, forgets to rebuild, and the
demo shows last week's UI. Timestamps cannot catch that -- a git checkout gives every file a
fresh, arbitrary mtime -- but contents can. `stamp` writes `ui/dist/build-manifest.json`, the
sha256 of every file the bundle is built from; `check` recomputes it and says which files moved.

Standard library only: the clean-clone test runs `check` before any venv exists.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "build-manifest.json"

#: Everything under these is an input to the bundle. `public/` is copied into dist verbatim.
INPUT_DIRS = ("ui/src", "ui/public")
INPUT_FILES = ("ui/index.html", "ui/package.json", "ui/package-lock.json",
               "ui/vite.config.ts", "ui/tsconfig.json")
IGNORED = {".DS_Store"}

#: A relative import in a .ts/.tsx file: `from "../x"`, `import "./x.css"`, `import("./x")`.
RELATIVE_IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*["'](\.{1,2}/[^"']+)["']""")
ASSET_REF = re.compile(r'(?:src|href)="(/[^/"][^"]*)"')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path, rel_dir: str) -> list[Path]:
    base = root / rel_dir
    return [p for p in sorted(base.rglob("*")) if p.is_file() and p.name not in IGNORED]


def inputs(root: Path = ROOT) -> dict[str, str]:
    """{repo-relative path: sha256} for every file the bundle is built from.

    That includes files *outside* ui/ that the source imports by relative path -- today
    `data/reference/nyc_modzcta.geojson`, which Map.tsx bundles as an asset. Found by reading
    the imports rather than listing them, so a new one cannot be forgotten.
    """
    found = [p for d in INPUT_DIRS for p in _files(root, d)]
    found += [root / f for f in INPUT_FILES if (root / f).is_file()]
    ui = (root / "ui").resolve()
    for src in _files(root, "ui/src"):
        if src.suffix not in {".ts", ".tsx"}:
            continue
        for spec in RELATIVE_IMPORT.findall(src.read_text(encoding="utf-8")):
            target = (src.parent / spec.split("?")[0]).resolve()
            if target.is_file() and ui not in target.parents:
                found.append(target)
    return {p.resolve().relative_to(root.resolve()).as_posix(): _digest(p) for p in found}


def problems(root: Path = ROOT) -> list[str]:
    """Every reason `ui/dist` cannot be trusted. Empty means it is present, whole and current."""
    dist = root / "ui" / "dist"
    index = dist / "index.html"
    if not index.is_file():
        return ["ui/dist/index.html is missing: there is no built UI to serve"]

    out = [f"index.html references {ref}, which is not in ui/dist"
           for ref in ASSET_REF.findall(index.read_text(encoding="utf-8"))
           if not (dist / ref.lstrip("/")).is_file()]

    manifest = dist / MANIFEST
    if not manifest.is_file():
        out.append(f"ui/dist/{MANIFEST} is missing, so nothing says what this bundle was built "
                   "from (an `npm run build` by hand deletes it)")
        return out

    built = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    now = inputs(root)
    for label, names in (("changed", [k for k in now if k in built and now[k] != built[k]]),
                         ("added  ", [k for k in now if k not in built]),
                         ("removed", [k for k in built if k not in now])):
        shown = ", ".join(sorted(names)[:8]) + (" ..." if len(names) > 8 else "")
        if names:
            out.append(f"{label} since ui/dist was built: {shown}")
    return out


def stamp(root: Path = ROOT) -> int:
    """Record what the current dist was built from. Run right after a successful build."""
    dist = root / "ui" / "dist"
    if not (dist / "index.html").is_file():
        print("ui/dist/index.html is missing; build first (`make ui`)", file=sys.stderr)
        return 1
    manifest = {"inputs": inputs(root)}
    (dist / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(f"ui/dist stamped: built from {len(manifest['inputs'])} source files")
    return 0


def check(root: Path = ROOT) -> int:
    found = problems(root)
    if not found:
        print(f"ui/dist is current ({len(inputs(root))} source files match the build)")
        return 0
    print("ui/dist cannot be trusted:", file=sys.stderr)
    for line in found:
        print(f"  - {line}", file=sys.stderr)
    print("Rebuild it and commit the result:  make ui", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    commands = {"stamp": stamp, "check": check}
    if len(argv) != 2 or argv[1] not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    return commands[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
