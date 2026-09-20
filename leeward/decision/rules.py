"""The hazard-triggered Act-now rules (SPEC §7.5), read from `act_now.yaml`.

    load(path)                                  -> {name: Rule}
    fired(vet_days, cohort, hazards, sites)     -> veteran_id, date, tier_rule

Each rule is one row of a table a clinician can read, and the table is the specification:
`tiers.assign()` has no Act-now mechanism of its own beyond what is written there. The four
rows encode things the posterior has not seen -- a closed station, an excluded drug class, a
stopped mail route, an outage in a home with nobody in it -- so a veteran who matches one is
Act-now regardless of where their probability sits.

The cells hold a deliberately small language: terms joined by `and` or `or` (never both in
one cell, so there is no precedence to get wrong), a term being a boolean column or
`column op value`. Small enough that the YAML reads as a table rather than as code, and
strict enough that `load()` rejects a name the schema does not have, an operator a column's
type cannot support, or a cell that mixes its connectives. Nobody reads diffs on this
project, so a rule that has quietly stopped firing is the failure worth designing against.

Every column is checked against `leeward/schema.py`, which is also how each cell knows which
frame it is about: `veteran` is the cohort, `station` is site_status, `zip` is hazards.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import yaml

from leeward import schema

PATH = Path(__file__).with_name("act_now.yaml")

#: The rules SPEC §7.5 names. `load()` refuses a file that does not hold exactly these, so
#: deleting a row from the YAML is a test failure rather than a silently quieter demo.
EXPECTED = (
    "site_dependent_at_a_closed_station",
    "controlled_substance_at_a_closed_station",
    "mail_order_supply_short_on_a_disrupted_day",
    "unattended_powered_equipment_in_an_outage",
)

#: Which cell of a row is about which contract table, and what it joins on.
FRAMES = {
    "veteran": ("cohort", ("veteran_id",)),
    "station": ("site_status", ("facility_id", "date")),
    "zip": ("hazards", ("modzcta", "date")),
}

#: The prose keys every row carries beside its conditions.
BECAUSE, SOURCE = "because", "source"

_OPS = {"==": operator.eq, "!=": operator.ne, "<=": operator.le,
        ">=": operator.ge, "<": operator.lt, ">": operator.gt}

#: Operators that only make sense on a number. A string column may only be == or !=.
_ORDERING = ("<=", ">=", "<", ">")


@dataclass(frozen=True)
class Rule:
    """One row: the cells that must all hold, the sentence, and where it comes from."""

    name: str
    when: dict[str, pl.Expr]      # cell key -> that cell's condition, already built
    because: str
    source: str


def load(path: Path | str | None = None) -> dict[str, Rule]:
    """Every rule in `act_now.yaml`, validated, in the order the file writes them."""
    path = Path(path) if path is not None else PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a mapping of rule name -> row")
    if set(raw) != set(EXPECTED):
        raise ValueError(
            f"{path}: must hold exactly the {len(EXPECTED)} rules of SPEC §7.5; missing "
            f"{sorted(set(EXPECTED) - set(raw))}, unknown {sorted(set(raw) - set(EXPECTED))}")

    out: dict[str, Rule] = {}
    for name, row in raw.items():
        if not isinstance(row, dict):
            raise ValueError(f"{path}: rule {name!r} must be a mapping of cell -> condition")
        unknown = set(row) - set(FRAMES) - {BECAUSE, SOURCE}
        if unknown:
            raise ValueError(f"{path}: rule {name!r} has cells {sorted(unknown)}; a row's "
                             f"cells are {sorted(FRAMES)}, plus {BECAUSE} and {SOURCE}")
        for key in (BECAUSE, SOURCE):
            text = row.get(key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: rule {name!r} needs a non-empty {key!r}. "
                                 f"{BECAUSE} is read aloud to a care team; {SOURCE} is how "
                                 f"anyone checks it.")
        when = {cell: _cell(path, name, cell, row[cell])
                for cell in FRAMES if cell in row}
        if not when:
            raise ValueError(f"{path}: rule {name!r} has no conditions, so it would make "
                             f"the whole panel Act-now")
        out[name] = Rule(name=name, when=when,
                         because=" ".join(row[BECAUSE].split()),
                         source=" ".join(row[SOURCE].split()))
    return out


def fired(vet_days: pl.DataFrame, cohort: pl.DataFrame, hazards: pl.DataFrame,
          site_status: pl.DataFrame, rules: dict[str, Rule] | None = None) -> pl.DataFrame:
    """`vet_days` with a `tier_rule` column: the first rule that matched, or null.

    Whatever columns `vet_days` arrived with are kept, so a caller that already has the
    frame it wants the answer on does not pay for a second join to get it back. Passed
    `veteran_id, date`, it returns exactly those two and `tier_rule`.

    `vet_days` supplies the (veteran, day) pairs to judge; the three tables supply the facts.

    Each cell is answered where its facts live -- the ZIP cells against the 5,000-row hazards
    table, the station cells against the 400-row site_status table, the veteran cells against
    the cohort -- and only the yes/no answers are joined onto the veteran-days. Joining the
    raw columns instead would drag every hazard reading across 10,000 veterans x a week, and
    this runs inside the capacity slider.

    A veteran-day whose ZIP or station has no row for that date reads as calm: a hazard
    nobody recorded is not a hazard, and a left join must never turn a missing row into a
    reason to call somebody.
    """
    table = rules if rules is not None else load()
    #: The cohort is joined first and carries the two keys the other joins need.
    sources = {"veteran": (cohort, ("modzcta", "facility_id")),
               "station": (site_status, ()), "zip": (hazards, ())}
    keep = vet_days.columns

    frame = vet_days.lazy()
    for cell, (_, keys) in FRAMES.items():
        source, carry = sources[cell]
        answers = [rule.when[cell].alias(_answer(name, cell))
                   for name, rule in table.items() if cell in rule.when]
        if not answers and not carry:
            continue
        frame = frame.join(
            source.lazy().select(*keys, *carry, *answers).unique(subset=keys),
            on=list(keys), how="left")

    match = pl.lit(None, dtype=pl.Utf8)
    for name, rule in reversed(list(table.items())):     # first written wins, so build back
        holds = pl.all_horizontal(
            *(pl.col(_answer(name, cell)) for cell in rule.when)).fill_null(False)
        match = pl.when(holds).then(pl.lit(name)).otherwise(match)
    # Lazy so the three joins, the four rule expressions and this projection fuse into one
    # plan: the intermediate frames carrying eight boolean columns across the whole panel
    # are never materialised. This runs inside the capacity slider.
    return frame.select(*keep, tier_rule=match).collect()


def _answer(rule: str, cell: str) -> str:
    return f"_{rule}__{cell}"


# --------------------------------------------------------------------------- #
# Reading one cell
# --------------------------------------------------------------------------- #

def _columns(cell: str) -> dict[str, pl.DataType]:
    name, _ = FRAMES[cell]
    return {c.name: c.dtype for c in schema.TABLES[name].columns}


def _cell(path: Path, rule: str, cell: str, text: object) -> pl.Expr:
    """One cell -> one expression. Terms joined by `and` or `or`, never both."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path}: {rule}.{cell} must be a condition, got {text!r}")
    has_and, has_or = " and " in f" {text} ", " or " in f" {text} "
    if has_and and has_or:
        raise ValueError(
            f"{path}: {rule}.{cell} mixes `and` with `or` ({text!r}). Use one or the other, "
            f"so the row means the same thing to everyone who reads it; split the rule in "
            f"two if it genuinely needs both.")
    joiner = " or " if has_or else " and "
    terms = [_term(path, rule, cell, part) for part in text.split(joiner)]
    return pl.any_horizontal(*terms) if has_or else pl.all_horizontal(*terms)


def _term(path: Path, rule: str, cell: str, text: str) -> pl.Expr:
    """`column`, or `column op value`."""
    parts = text.split()
    where = f"{path}: {rule}.{cell}"
    columns = _columns(cell)

    if len(parts) == 1:
        (name,) = parts
        dtype = _column(where, cell, name, columns)
        if dtype != pl.Boolean:
            raise ValueError(f"{where}: {name!r} is {dtype}, not a yes/no column, so it "
                             f"cannot stand on its own — write `{name} == <value>`")
        return pl.col(name)

    if len(parts) != 3:
        raise ValueError(f"{where}: {text!r} is not a condition. Write a yes/no column on "
                         f"its own, or `column op value` with op one of {sorted(_OPS)}.")
    name, op, literal = parts
    if op not in _OPS:
        raise ValueError(f"{where}: unknown operator {op!r} in {text!r}; "
                         f"operators are {sorted(_OPS)}")
    dtype = _column(where, cell, name, columns)
    col = pl.col(name)

    if dtype == pl.Boolean:
        raise ValueError(f"{where}: {name!r} is a yes/no column; write it on its own or "
                         f"negated, not as {text!r}")
    if dtype == pl.Utf8:
        if op in _ORDERING:
            raise ValueError(f"{where}: {name!r} holds words, not numbers, so {op!r} has no "
                             f"meaning for it; use == or !=")
        return _OPS[op](col, pl.lit(literal))
    try:
        value = float(literal)
    except ValueError:
        raise ValueError(f"{where}: {name!r} is {dtype}, so {literal!r} must be a "
                         f"number") from None
    return _OPS[op](col, pl.lit(value))


def _column(where: str, cell: str, name: str, columns: dict[str, pl.DataType]) -> pl.DataType:
    if name not in columns:
        table, _ = FRAMES[cell]
        raise ValueError(f"{where}: {name!r} is not a column of {table}. A rule may only "
                         f"read facts the contract actually carries.")
    return columns[name]
