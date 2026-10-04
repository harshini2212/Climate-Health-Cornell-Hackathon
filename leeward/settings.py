r"""Deployment settings, read once from `LEEWARD_*` environment variables.

Every default is the value that used to be hard-coded where it is read, so a clean clone with
no environment set behaves exactly as before. There is deliberately no `.env` file support:
`make demo` must work from a clean clone, and a stray `.env` silently changing the demo is the
failure that rule exists to prevent.

    LEEWARD_SCENARIO=sandy_then_heat
    LEEWARD_CORS_ORIGIN_REGEX='^http://(localhost|127\.0\.0\.1)(:\d+)?$'
    LEEWARD_OTP_STATIONS='["630", "630A4"]'

Two things are here rather than somewhere more obvious, and both are worth knowing:

* `data_root` is declared but **not yet read**. The data root is `schema.DATA`, and
  `leeward/schema.py` is the cohort lane's contract file; pointing it here is a one-line change
  for its owner. Until then, setting `LEEWARD_DATA_ROOT` changes nothing.
* `otp_stations` is a fact about facilities, not about a scenario: the cohort is built once
  and shared by every scenario, so it cannot live in one scenario's YAML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEEWARD_", frozen=True)

    #: The one scenario the cached hazards table serves. `make hazards` builds it from
    #: scenarios/sandy_then_heat.yaml, the demo's main scenario. The table does not record
    #: which scenario built it, so change this only together with what `make hazards` ran.
    scenario: str = "sandy_then_heat"

    #: Where the contract tables live. See the module docstring: not yet wired to `schema.DATA`.
    data_root: Path = ROOT / "data"

    #: Origins allowed to call the API cross-origin. The UI proxies /api in dev; this covers a
    #: build pointed straight at :8000 (VITE_API_URL).
    cors_origin_regex: str = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

    #: Sites that can dispense methadone for an opioid treatment program (docs/SPEC.md §5.2).
    otp_stations: tuple[str, ...] = ("630", "630A4")


@lru_cache(maxsize=1)
def get() -> Settings:
    return Settings()
