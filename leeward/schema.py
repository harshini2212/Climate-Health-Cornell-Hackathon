"""The frozen data contracts.

Every lane reads and writes these tables. Nothing else in the build is allowed to
invent a column name. If a lane needs a new column, the owner of this file adds it
here first and pushes; the lane then uses it.

Why this file is strict: nobody is reading the diffs. `validate()` is the only thing
standing between a plausible-looking agent change and a demo that shows wrong numbers,
so it checks dtypes, nullability, key uniqueness, value domains, and the `_synthetic`
flag convention -- not just that the columns exist.

Usage:

    from leeward import schema
    schema.validate(df, "cohort")          # raises SchemaError with a readable message
    df = schema.empty("scores")            # correctly-typed empty frame
    schema.NEEDS                           # the five needs, in canonical order
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REFERENCE = DATA / "reference"

# --------------------------------------------------------------------------- #
# Canonical vocabularies. Import these; never retype the strings.
# --------------------------------------------------------------------------- #

NEEDS = ["breathing", "heat", "mental", "treatment_gap", "access_loss"]

TIERS = ["act_now", "find_out", "self_serve", "everyday"]

BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

#: action -> the capacity bucket it consumes. `allocate.py` fills buckets, not actions.
ACTION_COST_UNIT = {
    "care_team_call": "call",
    "check_in_call": "call",
    "backup_power_plan": "call",
    "cold_chain_plan": "call",
    "early_refill": "refill",
    "switch_to_local_pickup": "refill",
    "cooling_center_ride": "ride",
    "clean_air_room": "ride",
    "alt_site_booking": "booking",
    "evacuation_assist": "evac",
    "assign_buddy": "partner_slot",
    "pharmacist_med_review": "pharmacist_slot",
    "controlled_substance_bridge": "va_fill",
    "heap_application": "free",
    "verified_text": "free",
}
ACTIONS = sorted(ACTION_COST_UNIT)

#: The care team's daily budget. The demo's capacity slider moves `call`.
DEFAULT_CAPACITY = {
    "call": 40, "refill": 200, "ride": 15, "booking": 20, "evac": 8,
    "partner_slot": 10, "pharmacist_slot": 12, "va_fill": 30, "free": 10_000,
}

POWERED_EQUIPMENT = ["none", "oxygen", "ventilator", "wheelchair", "bed", "dialysis_home"]
DEPLOYMENT_ERAS = ["vietnam", "gulf", "post911", "peacetime"]
CAREGIVER = ["none", "informal_coresident", "informal_remote", "va_pcafc"]
INCOME_BANDS = ["low", "mid", "high"]
FLOORS = ["basement", "ground", "upper"]
CHANNELS = ["VEText", "MHV", "care_team_phone"]
#: The fairness audit's strata. Census asks race and Hispanic origin as two questions, so
#: "Hispanic" is an origin of any race; "Other" folds AIAN, NHPI, some other race and multiracial.
RACES = ["White", "Black", "Asian", "Other"]
ETHNICITIES = ["Hispanic", "Non-Hispanic"]

#: Every outreach message must contain all of these. `test_guardrails.py` enforces it.
MANDATORY_MESSAGE_ELEMENTS = [
    "The VA will never ask you to pay, wire money, or share bank details",
    "833-388-7233",
    "988",
]


class SchemaError(AssertionError):
    """Raised when a table does not match its contract. Message names the table and column."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: pl.DataType
    doc: str = ""
    nullable: bool = False
    #: If set, every value must be in this set.
    values: tuple | None = None
    #: Closed interval every value must fall inside.
    bounds: tuple[float, float] | None = None
    #: Augmented (not read from a source): requires a `<name>_synthetic` boolean sibling.
    synthetic: bool = False


@dataclass(frozen=True)
class Table:
    name: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    doc: str = ""
    #: Columns a lane may add without touching this file (drivers, debug, joins).
    allow_extra: bool = True

    @property
    def path(self) -> Path:
        return DATA / f"{self.name}.parquet"

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


def _c(name, dtype, doc="", **kw) -> Column:
    return Column(name=name, dtype=dtype, doc=doc, **kw)


# --------------------------------------------------------------------------- #
# cohort.parquet -- one row per veteran
# --------------------------------------------------------------------------- #

_COHORT = Table(
    name="cohort",
    key=("veteran_id",),
    doc="The synthetic NYC veteran panel. Neighbourhood rates are real; people are not.",
    columns=(
        _c("veteran_id", pl.Utf8, "Synthea Patient.id"),
        _c("name_display", pl.Utf8, "Synthetic display name for the demo card", synthetic=True),
        _c("age", pl.Int32, bounds=(18, 110)),
        _c("sex", pl.Utf8, values=("M", "F")),
        _c("race", pl.Utf8, "Fairness audit only; drawn from the ZIP's ACS B03002 composition",
           nullable=True, values=tuple(RACES), synthetic=True),
        _c("ethnicity", pl.Utf8, "Fairness audit only; drawn jointly with race, same source",
           nullable=True, values=tuple(ETHNICITIES), synthetic=True),

        # geography -- modzcta is the join key everywhere
        _c("modzcta", pl.Utf8, "NYC Modified ZCTA, one of 178"),
        _c("borough", pl.Utf8, values=tuple(BOROUGHS)),
        _c("facility_id", pl.Utf8, "VHA station number, e.g. 630"),

        # health, from Synthea
        _c("copd", pl.Boolean), _c("asthma", pl.Boolean), _c("chf", pl.Boolean),
        _c("diabetes", pl.Boolean), _c("ckd_dialysis", pl.Boolean),
        _c("active_cancer_tx", pl.Boolean), _c("ptsd", pl.Boolean),
        _c("depression", pl.Boolean),
        _c("pact_presumptive", pl.Boolean, "Any PACT respiratory/cancer code"),
        _c("n_chronic", pl.Int32, bounds=(0, 30)),
        _c("er_visits_12m", pl.Int32, bounds=(0, 100)),
        _c("missed_refills_12m", pl.Int32, bounds=(0, 100)),
        _c("missed_appts_12m", pl.Int32, bounds=(0, 100)),

        # medication -- codes are real, supply and delivery are not
        _c("med_rxcuis", pl.List(pl.Utf8), "Active RxNorm codes from the record"),
        _c("va_drug_classes", pl.List(pl.Utf8), "Mapped via va_drug_class_members.parquet"),
        _c("n_active_meds", pl.Int32, bounds=(0, 60)),
        _c("med_thermoreg_score", pl.Float64, "Sum of heat-mechanism weights", bounds=(0, 30)),
        _c("acb_score", pl.Int32, "Anticholinergic burden, ACB scale; >=3 is meaningful",
           bounds=(0, 40)),
        _c("med_combo_raas_diuretic", pl.Boolean, "The combination CDC names explicitly"),
        _c("med_renal_triple", pl.Boolean, "NSAID on top of a diuretic and a RAAS agent"),
        _c("med_cold_chain", pl.Boolean, "Insulin and friends; drives the outage term"),
        _c("med_controlled", pl.Boolean, "Excluded from the VA retail emergency refill"),
        _c("med_narrow_ti", pl.Boolean, "Lithium, warfarin, levothyroxine, antiarrhythmics"),
        _c("mail_order_pharmacy", pl.Boolean, "~80% base rate, from VA's CMOP share",
           synthetic=True),
        _c("days_supply_remaining", pl.Int32, "30-day window / 90-day mail, uniform phase",
           bounds=(0, 120), synthetic=True),

        # exposure and housing -- no public per-person source, so all synthetic
        _c("deployment_era", pl.Utf8, values=tuple(DEPLOYMENT_ERAS), synthetic=True),
        _c("burn_pit_years", pl.Float64, "Latent truth; observed noisily via pact_presumptive",
           bounds=(0, 20), synthetic=True),
        _c("ptsd_severity", pl.Int32, bounds=(0, 4), synthetic=True),
        _c("home_ac", pl.Boolean, synthetic=True),
        _c("floor", pl.Utf8, values=tuple(FLOORS), synthetic=True),
        _c("on_methadone_otp", pl.Boolean, "Site-dependent treatment", synthetic=True),
        _c("powered_equipment", pl.Utf8, values=tuple(POWERED_EQUIPMENT), synthetic=True),

        # social and resources -- rates from CDC PLACES, assignment synthetic
        _c("lives_alone", pl.Boolean),
        _c("mobility_impaired", pl.Boolean, "Rate from PLACES mobility_crudeprev",
           synthetic=True),
        _c("caregiver", pl.Utf8, values=tuple(CAREGIVER), synthetic=True),
        _c("caregiver_contact_consent", pl.Boolean, synthetic=True),
        _c("income_band", pl.Utf8, values=tuple(INCOME_BANDS), synthetic=True),
        _c("low_assets", pl.Boolean, "Rate from PLACES shututility_crudeprev", synthetic=True),
        _c("transport_barrier", pl.Boolean, "Rate from PLACES lacktrpt_crudeprev",
           synthetic=True),

        # hazard exposure of the person's ZIP, joined from data/reference
        _c("evac_zone", pl.Int32, "0 = not in any zone", bounds=(0, 7)),
        _c("stormwater_flooded_frac", pl.Float64, bounds=(0, 1)),
        _c("hvi", pl.Int32, "Heat Vulnerability Index 1-5", bounds=(1, 5)),

        # consent flags for the partner export -- absence excludes a row, never redacts it
        _c("consent_partner_check", pl.Boolean, synthetic=True),
        _c("consent_ride", pl.Boolean, synthetic=True),
        _c("consent_housing", pl.Boolean, synthetic=True),
    ),
)

# --------------------------------------------------------------------------- #
# hazards.parquet -- one row per modzcta x day
# --------------------------------------------------------------------------- #

_HAZARDS = Table(
    name="hazards",
    key=("modzcta", "date"),
    doc="Daily hazard per ZIP, assembled from a scenario YAML plus data/reference.",
    columns=(
        _c("modzcta", pl.Utf8),
        _c("date", pl.Date),
        _c("heat_index_max_f", pl.Float64, bounds=(-20, 140)),
        _c("hot_day", pl.Boolean, "heat_index_max_f >= 82, NYC Health's own threshold"),
        _c("heat_alert", pl.Boolean),
        _c("pm25", pl.Float64, "ug/m3; replayed from AirNow for the smoke scenario",
           bounds=(0, 1000)),
        _c("smoke_alert", pl.Boolean),
        _c("flood_watch", pl.Boolean),
        _c("flood_warning", pl.Boolean),
        _c("flash_flood_emergency", pl.Boolean),
        _c("surge_ft", pl.Float64, bounds=(0, 30)),
        _c("evac_zone_ordered", pl.Int32, "Highest zone ordered to evacuate; 0 = none",
           bounds=(0, 7)),
        _c("floodnet_trip", pl.Boolean),
        _c("stormwater_flooded_frac", pl.Float64, bounds=(0, 1)),
        _c("outage_frac", pl.Float64, bounds=(0, 1)),
        _c("mail_delivery_disrupted", pl.Boolean,
           "Four in five VA prescriptions arrive by mail. Not a minor term."),
    ),
)

_SITE_STATUS = Table(
    name="site_status",
    key=("facility_id", "date"),
    doc="Per-facility daily closure. Separate table, not a dict column on hazards.",
    columns=(
        _c("facility_id", pl.Utf8),
        _c("date", pl.Date),
        _c("site_down", pl.Boolean),
        _c("evac_zone", pl.Int32, bounds=(0, 7)),
        _c("site_dependent_services", pl.Boolean, "Dialysis, infusion or OTP on site"),
    ),
)

# --------------------------------------------------------------------------- #
# outcomes / scores / actions / log
# --------------------------------------------------------------------------- #

_OUTCOMES = Table(
    name="outcomes",
    key=("veteran_id", "date", "need"),
    doc="Simulated ground truth from truth.json. Never shown in the UI.",
    columns=(
        _c("veteran_id", pl.Utf8),
        _c("date", pl.Date),
        _c("need", pl.Utf8, values=tuple(NEEDS)),
        _c("y", pl.Int32, bounds=(0, 1)),
    ),
)

_SCORES = Table(
    name="scores",
    key=("veteran_id", "date", "need"),
    doc="Posterior risk per veteran-day-need. Written by score.py at any model rung.",
    columns=(
        _c("veteran_id", pl.Utf8),
        _c("date", pl.Date),
        _c("need", pl.Utf8, values=tuple(NEEDS)),
        _c("p_mean", pl.Float64, bounds=(0, 1)),
        _c("p_lo80", pl.Float64, "10th percentile over posterior draws", bounds=(0, 1)),
        _c("p_hi80", pl.Float64, "90th percentile over posterior draws", bounds=(0, 1)),
        _c("p_epistemic_share", pl.Float64,
           "Var over draws / (Var over draws + mean p(1-p)). Drives the Find-out tier.",
           bounds=(0, 1)),
        _c("driver_1", pl.Utf8, "Plain-language phrase from posterior contributions",
           nullable=True),
        _c("driver_2", pl.Utf8, nullable=True),
        _c("driver_3", pl.Utf8, nullable=True),
        _c("driver_1_contrib", pl.Float64, "Log-odds contribution", nullable=True),
        _c("driver_2_contrib", pl.Float64, nullable=True),
        _c("driver_3_contrib", pl.Float64, nullable=True),
        _c("model_rung", pl.Int32, "Which ladder rung produced this. Say it on stage.",
           bounds=(0, 3)),
    ),
)

_ACTIONS = Table(
    name="actions",
    key=("date", "veteran_id", "action"),
    doc="A day's ranked action list, already cut at that day's capacity. `date` is the "
        "do-by day: the day the work happens, which is the risk day minus `lead_days`.",
    columns=(
        _c("action_id", pl.Utf8),
        _c("date", pl.Date, "The day this must be DONE by; risk day = date + lead_days"),
        _c("veteran_id", pl.Utf8),
        _c("action", pl.Utf8, values=tuple(ACTIONS)),
        _c("tier", pl.Utf8, values=tuple(TIERS)),
        _c("eha", pl.Float64, "Expected harm averted", bounds=(0, 1000)),
        _c("lead_days", pl.Int32, "Days ahead of the risk this must happen to work at all; "
                                  "from lead_days in leeward/decision/tau.yaml",
           bounds=(0, 30)),
        _c("rank", pl.Int32, bounds=(1, 1_000_000)),
        _c("capacity_bucket", pl.Utf8, values=tuple(sorted(set(ACTION_COST_UNIT.values())))),
        _c("rationale", pl.Utf8, "One sentence a care team member can read aloud"),
        _c("owner", pl.Utf8, "Who does it: care_team, pharmacist, partner, automated"),
        _c("message_id", pl.Utf8, nullable=True),
    ),
)

_OUTCOME_LOG = Table(
    name="outcome_log",
    key=("action_id",),
    doc="Append-only. 20-30 rows a week is what the Bayesian model absorbs without a retrain.",
    columns=(
        _c("action_id", pl.Utf8),
        _c("veteran_id", pl.Utf8),
        _c("date", pl.Date),
        _c("done", pl.Boolean),
        _c("reached", pl.Boolean),
        _c("need_occurred", pl.Boolean, nullable=True),
        _c("partner_ack", pl.Boolean, nullable=True),
        _c("logged_by", pl.Utf8),
    ),
)

TABLES: dict[str, Table] = {
    t.name: t for t in (_COHORT, _HAZARDS, _SITE_STATUS, _OUTCOMES,
                        _SCORES, _ACTIONS, _OUTCOME_LOG)
}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _fail(table: str, msg: str) -> None:
    raise SchemaError(f"[{table}] {msg}")


def validate(df: pl.DataFrame, table_name: str, *, check_keys: bool = True) -> pl.DataFrame:
    """Assert `df` satisfies the contract for `table_name`. Returns df so it can wrap a write.

    Checks, in order: table exists, no missing columns, no unexpected columns (if the table
    forbids them), dtype per column, nullability, value domain, numeric bounds, `_synthetic`
    siblings, and key uniqueness.
    """
    if table_name not in TABLES:
        _fail(table_name, f"unknown table; known tables are {sorted(TABLES)}")
    t = TABLES[table_name]
    have = set(df.columns)

    missing = [c.name for c in t.columns if c.name not in have]
    if missing:
        _fail(t.name, f"missing columns: {missing}")

    if not t.allow_extra:
        extra = sorted(have - {c.name for c in t.columns})
        if extra:
            _fail(t.name, f"unexpected columns: {extra}")

    for col in t.columns:
        s = df[col.name]

        if s.dtype != col.dtype:
            # Integer width is not worth failing a build over; anything else is.
            both_int = s.dtype.is_integer() and col.dtype.is_integer()
            both_float = s.dtype.is_float() and col.dtype.is_float()
            if not (both_int or both_float):
                _fail(t.name, f"{col.name}: dtype is {s.dtype}, contract says {col.dtype}")

        n_null = s.null_count()
        if n_null and not col.nullable:
            _fail(t.name, f"{col.name}: {n_null} nulls but the column is not nullable")

        if col.values is not None:
            bad = set(s.drop_nulls().unique().to_list()) - set(col.values)
            if bad:
                _fail(t.name, f"{col.name}: values {sorted(bad)[:5]} not in {list(col.values)}")

        if col.bounds is not None and s.dtype.is_numeric():
            lo, hi = col.bounds
            nn = s.drop_nulls()
            if nn.len() and (nn.min() < lo or nn.max() > hi):
                _fail(t.name,
                      f"{col.name}: range [{nn.min()}, {nn.max()}] outside contract [{lo}, {hi}]")

        if col.synthetic:
            flag = f"{col.name}_synthetic"
            if flag not in have:
                _fail(t.name,
                      f"{col.name} is synthetic and needs a boolean '{flag}' sibling. "
                      "Synthetic numbers have to say so -- that is the whole claim.")
            if not df[flag].fill_null(False).all():
                _fail(t.name, f"{flag} must be True for every row")

    if check_keys and t.key:
        dupes = df.height - df.select(list(t.key)).unique().height
        if dupes:
            _fail(t.name, f"{dupes} duplicate rows on key {t.key}")

    return df


def empty(table_name: str) -> pl.DataFrame:
    """A correctly-typed zero-row frame, including the `_synthetic` sibling columns."""
    t = TABLES[table_name]
    cols: dict[str, pl.Series] = {}
    for c in t.columns:
        cols[c.name] = pl.Series(c.name, [], dtype=c.dtype)
        if c.synthetic:
            cols[f"{c.name}_synthetic"] = pl.Series(f"{c.name}_synthetic", [], dtype=pl.Boolean)
    return pl.DataFrame(cols)


def write(df: pl.DataFrame, table_name: str) -> Path:
    """Validate then write to the canonical path. Use this instead of `write_parquet`."""
    validate(df, table_name)
    path = TABLES[table_name].path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read(table_name: str, *, validate_on_read: bool = True) -> pl.DataFrame:
    df = pl.read_parquet(TABLES[table_name].path)
    if validate_on_read:
        validate(df, table_name)
    return df


def describe(table_name: str) -> str:
    """Human-readable contract, for pasting into a prompt."""
    t = TABLES[table_name]
    lines = [f"{t.name}.parquet -- key {t.key}", f"  {t.doc}", ""]
    for c in t.columns:
        bits = [str(c.dtype)]
        if c.nullable:
            bits.append("nullable")
        if c.values:
            bits.append(f"in {list(c.values)}")
        if c.bounds:
            bits.append(f"in [{c.bounds[0]}, {c.bounds[1]}]")
        if c.synthetic:
            bits.append("SYNTHETIC (needs _synthetic sibling)")
        lines.append(f"  {c.name:28s} {', '.join(bits)}" + (f"  -- {c.doc}" if c.doc else ""))
    return "\n".join(lines)


if __name__ == "__main__":  # `python -m leeward.schema` prints every contract
    for name in TABLES:
        print(describe(name), end="\n\n")
