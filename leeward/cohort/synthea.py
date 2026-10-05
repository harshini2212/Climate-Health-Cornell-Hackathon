"""Synthea's veteran patients -> one profile row each -> a whole-patient draw for the cohort.

    python -m leeward.cohort.synthea               # what the committed profiles hold

Track A (heat with behavioral health and heat-sensitive medications) needs a veteran's
PTSD, depression, self-harm, substance use, cancer and homelessness to belong to the same
person as their prescriptions. The parametric cohort drew each from its own coin, so a
veteran with PTSD was no likelier than anyone else to be on the sertraline that PTSD gets
treated with. Synthea's veteran modules generate them together:

    veteran_ptsd, veteran_mdd, veteran_self_harm,          Synthea v4.0.0,
    veteran_substance_abuse_conditions/_treatment,         src/main/resources/modules/
    veteran_lung_cancer, veteran_prostate_cancer,
    dialysis, homelessness

**The run** is pinned below (RUN_ARGS): Synthea v4.0.0, jar sha256 JAR_SHA256, seed 0,
clinician seed 0, reference and end date 2026-01-01, 8,000 living patients aged 18-100 in
New York State, `generate.veteran_population_override=true` so every adult is simulated
with veteran prevalence. The four CSVs read are ~2 GB, in `data/raw/`, never committed.
`profiles_from_csv()` distils it to `data/reference/synthea_veteran_profiles.parquet`: one
row per patient, their sex and age, the Track A flags, and their active RxNorm codes.
That table is what a clean clone reads.

**The draw** gives each re-homed veteran one whole profile from their own sex x ACS age
band. Where they live, their age and sex still come from ACS B21001 (build.rehome); what
Synthea supplies is the person's clinical record, never their address.

Definitions, all at the reference date. Synthea's CSV exporter keeps 10 years of history
(`exporter.years_of_history`), so "in the record" means "in the last ten years":

    depression, homeless             a matching condition is open (no STOP)
    ptsd                             a PTSD diagnosis in the record. Synthea closes the
                                     condition when treatment ends; a VA problem list keeps it
    substance_use_disorder           an open alcohol, opioid or drug-use *disorder*
                                     (the "unhealthy drinking" screen finding is not one)
    suicide_risk                     an "at increased risk for suicide" or "suicidal thoughts"
                                     finding in the record. The run produced no suicide
                                     attempt by a living patient, so attempts cannot be a flag
    active_cancer_tx                 an open malignancy AND chemotherapy or radiotherapy in the
                                     365 days before the reference date, or an active
                                     antineoplastic. Synthea never closes its cancer care plan

**Weighting** (approved 2026-10-05). Synthea's veteran modules do not reproduce the published
rates: PTSD in ~1% of the record against VA's 5-15% past-year, substance use disorder in
~27% against NSDUH's 15-18%, and no ZIP-level depression gradient at all. So the draw is
tilted, per veteran, toward a cited rate for each of those three flags (`draw(targets=...)`):
profile j is chosen with probability proportional to exp(sum_k theta_k x_jk) inside the
veteran's own sex x age-band pool, with theta solved so every targeted flag hits that
veteran's target. Profiles stay whole -- the tilt only changes *which* real patient a
veteran gets -- so the medications still belong to the diagnoses. Cancer treatment,
suicide risk and homelessness take Synthea's own rates: no cited rate measures the same thing.

Two definitional gaps, stated rather than hidden: the PTSD target is *past-year* (VA NCPTSD)
while the flag is a diagnosis anywhere in the 10-year record, and PLACES depression is
*ever told* while the flag is an open condition. Both targets are the nearest published
rate, not an exact match. Cancer is held to the stricter rule and is not tilted, because
PLACES' ever-diagnosed cancer rate is three times any plausible in-treatment rate.

Thin pools: Synthea gives men now 65-74 almost no PTSD, so their PTSD carriers are
borrowed from ages 55-64 and 75+ (MIN_CARRIERS). With the committed profiles that is ~11
distinct records serving every PTSD veteran over 65; treat medication detail for that
group as thin.
"""

from __future__ import annotations

import logging
import zlib
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl

from leeward.schema import REFERENCE

log = logging.getLogger(__name__)

#: ACS B21001 age bands -> inclusive age range. The 75+ upper bound is open. build.py
#: re-homes on these bands and this module bootstraps within them, so they live here once.
BANDS: dict[str, tuple[int, int | None]] = {
    "18_34": (18, 34), "35_54": (35, 54), "55_64": (55, 64),
    "65_74": (65, 74), "75plus": (75, None)}

SYNTHEA_VERSION = "v4.0.0"
JAR = "synthea-with-dependencies.jar"
JAR_URL = (f"https://github.com/synthetichealth/synthea/releases/download/"
           f"{SYNTHEA_VERSION}/{JAR}")
JAR_SHA256 = "ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1"
REFERENCE_DATE = date(2026, 1, 1)
POPULATION = 8_000
#: Exactly the arguments the committed profiles were generated with, after `java -jar JAR`
#: (argv tokens). The fetcher adds only output settings: `--exporter.baseDirectory` and
#: `--exporter.csv.included_files`.
RUN_ARGS = (
    "-s", "0", "-cs", "0", "-r", "20260101", "-e", "20260101", "-p", str(POPULATION),
    "-a", "18-100", "--generate.veteran_population_override=true",
    "--exporter.csv.export=true", "--exporter.fhir.export=false",
    "--exporter.hospital.fhir.export=false", "--exporter.practitioner.fhir.export=false",
    "New York",
)

#: The CSV files the distiller reads; the fetcher asks Synthea for these alone.
CSV_FILES = ("patients.csv", "conditions.csv", "medications.csv", "procedures.csv")

#: Track A flag -> the condition codes that set it, read off the Synthea v4.0.0 module
#: files named alongside. Codes only: the CSV's SYSTEM column says SNOMED-CT for all of these.
CONDITION_CODES: dict[str, frozenset[str]] = {
    # veteran_ptsd.json
    "ptsd": frozenset({"47505003"}),
    # veteran_mdd.json: single episode, and the recurrent disorder
    "depression": frozenset({"36923009", "370143000"}),
    # homelessness.json
    "homeless": frozenset({"32911000"}),
    # veteran_substance_abuse_treatment.json (alcoholism, opioid abuse) and the opioid and
    # drug-use states it hands on to (dependent drug abuse, misuses drugs). Smoking is not a
    # use disorder here, and "unhealthy alcohol drinking behavior" is a screening finding.
    "substance_use_disorder": frozenset({"7200002", "5602001", "6525002", "361055000"}),
    # veteran_ptsd.json and veteran_self_harm.json: "At increased risk for suicide (finding)"
    # and "Suicidal thoughts (finding)". The attempt states (86849004, 287185009, 287182007)
    # never occurred in a living patient of this run, and completed suicides are dead.
    "suicide_risk": frozenset({"225444004", "6471006"}),
}
#: Flags that count a condition whatever its STOP date, inside the exported history.
EVER = frozenset({"ptsd", "suicide_risk"})

#: A cancer is being treated when one of these procedures fell in the last CANCER_TX_DAYS...
#: Every chemotherapy and radiotherapy procedure the v4.0.0 cancer modules emit (SNOMED;
#: the run's dental "radiation shield" and X-ray D-codes are not treatment).
CANCER_TX_PROCEDURES = frozenset({
    "703423002",    # Combined chemotherapy and radiation therapy
    "367336001",    # Chemotherapy
    "394894008",    # Pre-operative chemotherapy
    "33195004",     # External beam radiation therapy
    "1287742003",   # Radiotherapy
    "385798007",    # Radiation therapy care
    "384692006",    # Intracavitary brachytherapy
    "447759004",    # Brachytherapy of breast
})
CANCER_TX_DAYS = 365
#: ...or the patient has an active medication in an antineoplastic VA class.
ANTINEOPLASTIC_CLASS_PREFIX = "AN"

FLAGS = (*CONDITION_CODES, "active_cancer_tx")
PROFILE_SCHEMA = {"profile_id": pl.Utf8, "age": pl.Int32, "sex": pl.Utf8,
                  **{f: pl.Boolean for f in FLAGS}, "rxcuis": pl.List(pl.Utf8)}


def band_expr(age: pl.Expr) -> pl.Expr:
    """The ACS band an age falls in, as a polars expression."""
    out = pl.lit(None, dtype=pl.Utf8)
    for name, (lo, _) in BANDS.items():
        out = pl.when(age >= lo).then(pl.lit(name)).otherwise(out)
    return out


def band_of(age: np.ndarray) -> np.ndarray:
    lows = np.array([lo for lo, _ in BANDS.values()])
    names = np.array(list(BANDS))
    return names[np.searchsorted(lows, age, side="right") - 1]


# --------------------------------------------------------------------------- #
# Distil: Synthea CSV export -> one row per living adult
# --------------------------------------------------------------------------- #

def _read(csv_dir: Path, name: str, cols: list[str]) -> pl.DataFrame:
    return pl.read_csv(csv_dir / f"{name}.csv", columns=cols, infer_schema=False)


def _open_at(df: pl.DataFrame, ref: date) -> pl.DataFrame:
    """Rows that started on or before `ref` and had not stopped by it."""
    start = pl.col("START").str.slice(0, 10).str.to_date()
    stop = pl.col("STOP").str.slice(0, 10).str.to_date()
    return df.filter((start <= ref) & (stop.is_null() | (stop > ref)))


def _malignant(desc: pl.Expr, code: pl.Expr) -> pl.Expr:
    """A cancer diagnosis, not a suspicion and not a benign growth. Synthea's head-and-neck
    and other cancer modules write ICD-10 C-codes; the rest write SNOMED with a description."""
    d = desc.str.to_lowercase()
    snomed = (d.str.contains("malignant|carcinoma|cancer|neoplasm of prostate|lymphoma|leukemia")
              & ~d.str.contains("suspected|benign|history of|screening"))
    return snomed | code.str.contains(r"^C\d")


def profiles_from_csv(csv_dir: Path, ref: date = REFERENCE_DATE,
                      antineoplastic_rxcuis: frozenset[str] | None = None,
                      cancer_tx_procedures: frozenset[str] | set[str] | None = None,
                      ) -> pl.DataFrame:
    """Distil a Synthea CSV export to the profile table. Dead and under-18 patients drop."""
    if antineoplastic_rxcuis is None:
        antineoplastic_rxcuis = _antineoplastic_rxcuis()
    if cancer_tx_procedures is None:
        cancer_tx_procedures = CANCER_TX_PROCEDURES
    patients = (_read(csv_dir, "patients", ["Id", "BIRTHDATE", "DEATHDATE", "GENDER"])
                .filter(pl.col("DEATHDATE").is_null() | (pl.col("DEATHDATE") == ""))
                .with_columns(pl.col("BIRTHDATE").str.to_date().alias("birth")))
    patients = (patients.with_columns(pl.Series("age", [_years(b, ref) for b in patients["birth"]],
                                                dtype=pl.Int32))
                .filter(pl.col("age") >= 18))

    conds = _read(csv_dir, "conditions", ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])
    open_conds = _open_at(conds, ref)
    started = conds.filter(pl.col("START").str.slice(0, 10).str.to_date() <= ref)

    flags = []
    for field, codes in CONDITION_CODES.items():
        src = started if field in EVER else open_conds
        flags.append(src.filter(pl.col("CODE").is_in(list(codes)))
                        .select(pl.col("PATIENT").unique()).with_columns(pl.lit(True).alias(field)))

    meds = _open_at(_read(csv_dir, "medications", ["START", "STOP", "PATIENT", "CODE"]), ref)
    rx = meds.group_by("PATIENT").agg(pl.col("CODE").unique().sort().alias("rxcuis"))

    malignant = (open_conds.filter(_malignant(pl.col("DESCRIPTION"), pl.col("CODE")))
                 .select(pl.col("PATIENT").unique()))
    procs = _read(csv_dir, "procedures", ["START", "PATIENT", "CODE"])
    when = pl.col("START").str.slice(0, 10).str.to_date()
    recent = procs.filter((when <= ref) & (when > ref - timedelta(days=CANCER_TX_DAYS)))
    treated = pl.concat([
        recent.filter(pl.col("CODE").is_in(list(cancer_tx_procedures))).select("PATIENT"),
        meds.filter(pl.col("CODE").is_in(list(antineoplastic_rxcuis))).select("PATIENT"),
    ]).unique()
    cancer = malignant.join(treated, on="PATIENT").with_columns(
        pl.lit(True).alias("active_cancer_tx"))

    out = patients.select(pl.col("Id").alias("PATIENT"), "age",
                          pl.col("GENDER").alias("sex"))
    for f in [*flags, cancer]:
        out = out.join(f, on="PATIENT", how="left")
    out = (out.join(rx, on="PATIENT", how="left")
              .with_columns(*[pl.col(f).fill_null(False) for f in FLAGS],
                            pl.col("rxcuis").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))))
              .rename({"PATIENT": "profile_id"}))
    return out.select(list(PROFILE_SCHEMA)).cast(PROFILE_SCHEMA).sort("profile_id")  # type: ignore[arg-type]


def _years(birth: date, ref: date) -> int:
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def _antineoplastic_rxcuis() -> frozenset[str]:
    xw = pl.read_parquet(REFERENCE / "va_drug_class_members.parquet")
    return frozenset(xw.filter(pl.col("va_class_id").str.starts_with(
        ANTINEOPLASTIC_CLASS_PREFIX))["rxcui"].to_list())


# --------------------------------------------------------------------------- #
# Draw: one whole profile per veteran, within sex x age band
# --------------------------------------------------------------------------- #

@cache
def load_profiles() -> pl.DataFrame:
    return pl.read_parquet(REFERENCE / "synthea_veteran_profiles.parquet")


#: A pool with fewer carriers of a targeted flag than this borrows that flag's carriers
#: from the nearest band of the same sex (`_pool`). Synthea makes men now 65-74 peacetime
#: veterans with almost no PTSD; without this their cited rate is unreachable.
MIN_CARRIERS = 5

#: Newton steps for the tilt, and the cap on |theta|: a flag no profile in the pool carries
#: cannot be reached by reweighting, and an unbounded theta would only hide that.
TILT_STEPS, TILT_CAP = 25, 8.0


def tilt(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Selection weights over a pool, one row per veteran: w_ij ∝ exp(theta_i . x_j).

    `x` is the pool's (m, k) 0/1 flag matrix and `target` the (n, k) rate each veteran's
    draw should carry. theta is solved coordinate-wise on the logit scale, which converges
    in a handful of steps for k <= 3. Returns row-normalised (n, m) probabilities.
    """
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    t = np.clip(target, 1e-4, 1 - 1e-4)
    theta = np.zeros_like(t)
    for _ in range(TILT_STEPS):
        for k in range(x.shape[1]):
            w = np.exp(np.clip(theta @ x.T, -50, 50))
            p = np.clip((w @ x[:, k]) / w.sum(axis=1), 1e-6, 1 - 1e-6)
            theta[:, k] = np.clip(theta[:, k] + logit(t[:, k]) - logit(p), -TILT_CAP, TILT_CAP)
    w = np.exp(np.clip(theta @ x.T, -50, 50))
    return w / w.sum(axis=1, keepdims=True)


def _pool(pools: dict, key: tuple[str, str], x: np.ndarray | None, flags: list[str]
          ) -> np.ndarray:
    """The veteran's own sex x band pool, plus -- for each targeted flag it holds fewer than
    MIN_CARRIERS of -- that flag's carriers from the nearest bands of the same sex, one
    ring at a time. Only carriers are borrowed, so everyone the tilt does not push toward
    the flag still gets a record from their own band."""
    pool = pools[key]
    if x is None:
        return pool
    names = list(BANDS)
    here = names.index(key[1])
    extra: list[np.ndarray] = []
    for k in range(len(flags)):
        have = int(x[pool, k].sum())
        for ring in range(1, len(names)):
            if have >= MIN_CARRIERS:
                break
            for b in (here - ring, here + ring):
                if 0 <= b < len(names) and (key[0], names[b]) in pools:
                    other = pools[(key[0], names[b])]
                    carriers = other[x[other, k] > 0]
                    extra.append(carriers)
                    have += carriers.size
    return np.unique(np.concatenate([pool, *extra])) if extra else pool


def _warn_if_short(key: tuple[str, str], flags: list[str], got: np.ndarray,
                   want: np.ndarray, tol: float = 0.005) -> None:
    """Say so when a target is out of reach even after borrowing: theta sits at TILT_CAP and
    the rate falls short. Silence here would read as calibrated."""
    gap = np.abs(got - want).max(axis=0)
    for k, f in enumerate(flags):
        if gap[k] > tol:
            log.warning("synthea draw: %s in %s/%s misses its cited rate by up to %.3f; "
                        "too few carriers in the profiles", f, key[0], key[1], gap[k])


def draw(sex: np.ndarray, age: np.ndarray, seed: int,
         targets: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """One profile row per veteran, in the order given, from the veteran's own sex and band.

    Whole profiles, never field by field: that is what keeps a diagnosis with the
    prescription that treats it. `targets` maps a flag to each veteran's cited rate; the
    pick is then tilted (see `tilt`) so those flags land on their rates. Without it, the
    draw is uniform and every flag takes Synthea's own rate.
    """
    targets = targets or {}
    flags = list(targets)
    profiles = load_profiles().with_columns(band_expr(pl.col("age")).alias("band"))
    pools = {k: g["_i"].to_numpy() for k, g in
             profiles.with_row_index("_i").group_by("sex", "band", maintain_order=True)}
    x_all = profiles.select(flags).to_numpy().astype(float) if flags else None
    t_all = np.column_stack([targets[f] for f in flags]) if flags else None
    bands = band_of(age)
    u = np.random.default_rng([seed, zlib.crc32(b"synthea_profile")]).random(len(sex))
    pick = np.empty(len(sex), dtype=np.int64)
    for key in sorted(set(zip(sex.tolist(), bands.tolist(), strict=True))):
        if key not in pools:
            raise SystemExit(f"no Synthea profile for sex {key[0]}, band {key[1]}; "
                             "synthea_veteran_profiles.parquet is incomplete")
        pool = _pool(pools, key, x_all, flags)
        rows = np.flatnonzero((sex == key[0]) & (bands == key[1]))
        if not flags:
            pick[rows] = pool[(u[rows] * len(pool)).astype(np.int64)]
            continue
        # Targets repeat (one per ZIP x era), so solve the tilt once per distinct row.
        uniq, inv = np.unique(t_all[rows], axis=0, return_inverse=True)  # type: ignore[index]
        x = x_all[pool]                                                  # type: ignore[index]
        prob = tilt(x, uniq)
        _warn_if_short(key, flags, prob @ x, uniq)
        cdf = np.cumsum(prob, axis=1)[inv.reshape(-1)]
        j = np.minimum((cdf < u[rows, None] * cdf[:, -1:]).sum(axis=1), len(pool) - 1)
        pick[rows] = pool[j]
    return profiles[pick].drop("band")


def main() -> int:
    p = load_profiles()
    print(f"{p.height:,} Synthea {SYNTHEA_VERSION} veteran profiles "
          f"(seed 0, reference {REFERENCE_DATE})\n")
    for f in FLAGS:
        print(f"  {f:24s} {p[f].sum() / p.height:6.2%}")
    print(f"  {'median active meds':24s} "
          f"{np.median(p['rxcuis'].list.len().to_numpy()):6.0f}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
