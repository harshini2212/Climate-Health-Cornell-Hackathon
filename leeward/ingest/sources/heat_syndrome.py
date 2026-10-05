"""NYC Health's daily heat-syndrome emergency department visits, May-September 2017-2021.

    https://github.com/nychealth/heat-syndrome-data   (Apache-2.0)

Two CSVs, one row per day, citywide:

* `edheat1720_supp.csv` -- 2017-2020, dates as `7/1/2020`, columns date, temp, count.
* `edheat2021_live.csv` -- 2021, dates as `2021-07-01`, columns date, count, temp. The
  publisher calls 2021 "live": recent counts may still be revised.

`MAX_DAILY_TEMP` is the higher of the heat index and the air temperature at the National
Weather Service's LaGuardia station -- the same quantity `design.py` hinges at 82 F. Counts
are syndromic (chief complaint or a heat diagnosis code), so they are suspected heat illness,
not confirmed, and miss residents treated outside the city. This is the real outcome series
`leeward/eval/backtest.py` scores the heat model against.
"""

from __future__ import annotations

import io
from collections.abc import Mapping

import polars as pl
import requests

REPO = "https://github.com/nychealth/heat-syndrome-data"
RAW = "https://raw.githubusercontent.com/nychealth/heat-syndrome-data/master/"
FILES = ("edheat1720_supp.csv", "edheat2021_live.csv")
#: Every row carries this tag, and the back-test refuses a frame without it.
SOURCE = "nychealth/heat-syndrome-data"
STEM = "nyc_heat_ed_daily"

_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d")


def parse(texts: Mapping[str, str]) -> pl.DataFrame:
    """{filename: csv text} -> one row per day: date, max_temp_f, heat_ed_visits, file, source."""
    frames = []
    for name, text in texts.items():
        raw = pl.read_csv(io.StringIO(text), infer_schema=False)
        frames.append(raw.select(
            date=pl.coalesce(*(pl.col("END_DATE").str.to_date(f, strict=False)
                               for f in _DATE_FORMATS)),
            max_temp_f=pl.col("MAX_DAILY_TEMP").cast(pl.Float64),
            heat_ed_visits=pl.col("HEAT_ED_VISIT_COUNT").cast(pl.Int64),
            file=pl.lit(name),
        ))
    df = pl.concat(frames).with_columns(source=pl.lit(SOURCE)).sort("date")

    if df.null_count().sum_horizontal().item():
        raise ValueError(f"unparsed values in {list(texts)}: {df.null_count().to_dicts()[0]}")
    dup = df.filter(pl.col("date").is_duplicated())["date"].unique()
    if dup.len():
        raise ValueError(f"{dup.len()} days published more than once, e.g. {dup[0]}")
    if (df["heat_ed_visits"] < 0).any():
        raise ValueError("negative heat ED visit count")
    return df


def fetch() -> pl.DataFrame:
    texts = {}
    for name in FILES:
        r = requests.get(RAW + name, timeout=60)
        r.raise_for_status()
        texts[name] = r.text
    return parse(texts)
