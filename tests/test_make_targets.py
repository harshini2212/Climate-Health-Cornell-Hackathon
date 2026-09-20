"""The Makefile is documentation that executes, so the gate holds it to both jobs.

`README.md`, `docs/PROMPTS.md` and `docs/BUILD_PLAN.md` tell people to run make targets;
the Makefile is the one copy of that instruction which can be checked by machine. When a
target names a module nobody wrote, the person following the docs finds out at 2am from
`No module named`, which does not say whether the module was renamed, never written, or is
theirs to write. These tests catch that at the gate instead.

Nothing here runs the pipeline. `make cohort` takes minutes, `make sources` needs the
network and `make demo` blocks on two servers. `make -n` prints exactly the command lines
make would run -- variables expanded, conditionals resolved -- and the command line is the
part that goes stale. The single exception is `make fit`, which is run for real *only*
while it is a refusal, because a refusal is all it costs.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: `make setup` is the target that creates the environment, so it is the one recipe whose
#: command lines cannot assume an environment exists -- it names `python -m pip`, and a uv
#: venv has no pip in it until this target has run.
BOOTSTRAP = {"setup"}

#: `$(PY)` expands to `.venv/bin/python`, CI's is plain `python`, and `make setup` falls
#: back to `python3.11`. Match the interpreter rather than any one spelling of it.
INTERPRETER = re.compile(r"(?:^|/)python[0-9.]*$")


def documented_targets() -> list[str]:
    """Every target carrying a `##` comment -- i.e. everything `make help` offers."""
    src = (ROOT / "Makefile").read_text(encoding="utf-8")
    found = {m.group(1) for m in re.finditer(r"^([a-z][\w-]*):.*?##", src, re.MULTILINE)}
    assert len(found) > 10, f"only found {sorted(found)}; the `##` help convention has changed"
    return sorted(found - BOOTSTRAP)


def recipe(target: str) -> str:
    """The command lines make would run for `target`, having run none of them."""
    p = subprocess.run(["make", "-n", target], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, f"`make -n {target}` does not even parse:\n{p.stderr}"
    return p.stdout


def _tokens(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:          # an apostrophe inside an echo, say
        return text.split()


def reads(text: str) -> tuple[list[str], list[str], list[str]]:
    """(modules, app specs, script paths) named by a recipe.

    Modules are what follows `python -m`. App specs are the `package.module:attr` argument
    handed to uvicorn, which is the other way `make demo` can name something that is not
    there. Script paths are the `.py` and `.sh` files a recipe runs directly.
    """
    modules: list[str] = []
    apps: list[str] = []
    scripts: list[str] = []
    toks = _tokens(text)
    for i, tok in enumerate(toks):
        if tok == "bash" and i + 1 < len(toks) and toks[i + 1].endswith(".sh"):
            scripts.append(toks[i + 1])
        if not INTERPRETER.search(tok):
            continue
        j = i + 1
        while j < len(toks) and toks[j].startswith("-") and toks[j] != "-m":
            j += 1                                          # -u, -O and friends
        if j + 1 < len(toks) and toks[j] == "-m":
            modules.append(toks[j + 1])
            if toks[j + 1] == "uvicorn" and j + 2 < len(toks) and ":" in toks[j + 2]:
                apps.append(toks[j + 2])
        elif j < len(toks) and toks[j].endswith(".py"):
            scripts.append(toks[j])
    return modules, apps, scripts


@pytest.mark.parametrize("target", documented_targets())
def test_every_documented_target_names_something_that_exists(target: str) -> None:
    """A documented target that dies on `No module named` is a broken instruction.

    `find_spec` locates the module without executing it: the failure this guards against is
    a Makefile naming a module that was renamed or never written, and importing half the
    pipeline to prove it would make the gate slow and give it new ways to go red.
    """
    modules, _, scripts = reads(recipe(target))
    for module in modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        assert found, (
            f"`make {target}` runs `python -m {module}`, which does not exist. Point the "
            f"target at a module that does, or make it fail with a sentence saying why.")
    for script in scripts:
        assert (ROOT / script).exists(), f"`make {target}` runs {script}, which is not there"


def test_make_score_runs_the_scorer_that_exists_and_then_allocates() -> None:
    """Scoring is two stages, and `scripts/baseline.py` is where that stage list lives.

    The Makefile and the baseline recorder disagreeing about what scoring means is how a
    recorded baseline ends up describing a pipeline nobody ran. `score.py` -- the posterior
    scorer -- takes over from `score_prior.py` the moment it lands, with no edit here.
    """
    scorer = ("leeward.model.score" if (ROOT / "leeward/model/score.py").exists()
              else "leeward.model.score_prior")
    modules, _, _ = reads(recipe("score"))
    assert modules == [scorer, "leeward.decision.allocate"], (
        f"`make score` runs {modules}; the stages that exist are "
        f"[{scorer!r}, 'leeward.decision.allocate'] (see scripts/baseline.py STAGES)")


@pytest.mark.skipif((ROOT / "leeward/model/fit.py").exists(),
                    reason="leeward/model/fit.py landed -- `make fit` fits for real now, and "
                           "test_every_documented_target_names_something_that_exists covers it")
def test_make_fit_explains_itself_while_there_is_nothing_to_fit() -> None:
    """Failing is fine. Failing without saying why is what costs somebody their night.

    This one target is run for real, because while there is no fitter the whole recipe is
    an echo and an exit. When `leeward/model/fit.py` lands this test skips itself rather
    than starting a ten-minute NUTS run inside the gate.
    """
    p = subprocess.run(["make", "fit"], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    assert p.returncode != 0, "`make fit` cannot report success while there is no fit to run"
    assert "No module named" not in out, (
        "`make fit` fails with an import error. Name the rung that is built and the module "
        f"that would have to land instead:\n{out}")
    assert "rung 0" in out.lower(), f"`make fit` does not say which rung is built:\n{out}"
    assert "leeward/model/fit.py" in out, (
        f"`make fit` does not name the module that would have to land:\n{out}")


def test_make_demo_serves_an_app_that_exists() -> None:
    """`make demo` blocks, so it is never run here -- but what it boots is checked.

    A typo in the `module:attr` uvicorn is handed, or a renamed npm script, surfaces as a
    dead port thirty seconds before a demo. Both are cheap to check from here.
    """
    text = recipe("demo")
    modules, apps, _ = reads(text)
    assert "uvicorn" in modules, "`make demo` no longer boots the API through uvicorn"
    assert apps, "`make demo` hands uvicorn no `module:attr` app to serve"
    for spec in apps:
        name, _, attr = spec.partition(":")
        module = importlib.import_module(name)
        assert getattr(module, attr, None) is not None, (
            f"`make demo` serves {spec}, but {name} has no {attr!r}; the API would not boot")

    package = json.loads((ROOT / "ui/package.json").read_text(encoding="utf-8"))
    for script in re.findall(r"npm run (\S+)", text):
        assert script in package.get("scripts", {}), (
            f"`make demo` runs `npm run {script}`, which ui/package.json does not define")
