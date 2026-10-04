# Contributing to Leeward

## Setup

```bash
git clone https://github.com/harshini2212/Climate-Health-Cornell-Hackathon.git
cd Climate-Health-Cornell-Hackathon
make setup       # venv + locked dependencies
pre-commit install
make check       # lint + every test, in parallel
make demo        # API + UI, offline, no .env
```

Python 3.11 (`.python-version`) and the Node version in `ui/.nvmrc`.

## How work lands

1. **Branch from `main`** with a short prefix: `feat/`, `fix/`, `data/`, `model/`, `docs/`,
   `cleanup/`. One change per branch.
2. **Write the test first.** For a bug, a failing test that reproduces it; for a feature, the
   acceptance test. If a skipping guardrail already covers your module, it is your spec.
3. **`make check` green** before you open the PR. Run `make perf` too if you touched the
   allocator, the scorer or the API.
4. **Open a PR** that says what changed, why, and how you verified it (the commands you ran
   and what they returned). Link the issue if there is one.
5. **One review**, then squash-merge. Reviewers ask whether the change improves the codebase,
   not whether it is perfect; prefix optional comments with `Nit:`.

Parallel work in several git worktrees is fine. Split it by files, so two branches never edit
the same file, and agree the merge order up front.

## Changes that need more than one review

- **Contracts** (`leeward/schema.py`, `leeward/api/schemas.py`, `leeward/contracts/`,
  `leeward/model/design.py`): edit `docs/SPEC.md` §3 in the same PR and say so in the
  description.
- **Anything touching data classification or PHI**: see the rule in `CLAUDE.md`.
- **Outreach message rules** and the guardrails that enforce them.
- **New dependencies**: say what they cost (size, licence, maintenance) in the PR.

## Data and numbers

- Every real number shown anywhere is in `docs/sources.md` with a URL and the date it was read.
- Synthetic values carry a `_synthetic` flag and say so on screen.
- Reference tables are fetched by `scripts/fetch_sources.py` and hashed in
  `data/reference/manifest.json`; `tests/guardrails/test_reference.py` checks the hashes.

## Where things are

| Path | What |
| --- | --- |
| `leeward/` | The package: `ingest`, `cohort` (synthetic generator), `model`, `decision`, `outreach`, `api`, `eval` |
| `ui/` | React + Vite + deck.gl |
| `tests/` | `unit`, `contracts`, `guardrails`, `demo` |
| `docs/SPEC.md` | Contracts and design |
| `docs/ROADMAP.md` | What comes next |
