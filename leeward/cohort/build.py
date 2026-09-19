"""The synthetic NYC veteran panel -> `data/cohort.parquet`.

    python -m leeward.cohort.build                  # 10,000 veterans, seed 0
    python -m leeward.cohort.build --n 2000 --seed 3

Synthetic people, real places. Where each veteran lives, their age band and their sex are
drawn jointly from ACS B21001, and every neighbourhood rate they are drawn against is read
from `data/reference/`, never typed in:

    where, age band, sex         acs_veterans_by_zcta       P(modzcta, sex, band) ∝ veterans
    mobility, low assets,        places_zcta_nyc            CDC PLACES crude prevalence,
      transport, no caregiver,                              per ZIP (see PLACES_RATES)
      COPD, asthma, cancer,
      depression, diabetes, CHF,
      low income
    powered equipment, dialysis  empower_ny_zip ÷ ACS 65+   HHS emPOWER, per ZIP
    evac zone, stormwater, HVI   evac_zone_by_modzcta, stormwater_by_modzcta, hvi_by_zcta
    facility                     va_facilities_nyc_hazard   nearest care site (see below)
    PTSD                         VA National Center for PTSD, past-year rate by era

What no public source publishes per person or per ZIP is drawn from the named constants in
the ASSUMPTIONS block, and every column those feed carries a `_synthetic` flag.

This is the *parametric* cohort. The VA Synthea release is a 4 GB CSV (data/README.md), so
until a Synthea reader lands, the health columns Synthea would supply are drawn here. The
medication columns are not drawn at all: `leeward/cohort/medications.py` gives each veteran
a real active-medication list from Synthea's public sample and derives every flag from the
VA drug class and CDC mechanism it maps to. `race` and `ethnicity` are null. Nothing in
`data/reference/` gives a veteran's race, and inventing one would give the fairness audit a
stratum that means nothing.

Facility: the nearest VA care site *in the veteran's own borough* by haversine. Straight
lines cross water in New York, and would send half of Staten Island to Brooklyn. Dialysis
goes to the nearest site that offers site-dependent care, and methadone OTP goes to
Manhattan or Brooklyn (docs/SPEC.md §5.2), wherever the veteran lives.
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np
import polars as pl

from leeward import schema
from leeward.cohort import medications
from leeward.schema import REFERENCE

N_DEFAULT = 10_000

#: ACS B21001 age bands -> inclusive age range. The 75+ upper bound is MAX_AGE.
BANDS = {"18_34": (18, 34), "35_54": (35, 54), "55_64": (55, 64),
         "65_74": (65, 74), "75plus": (75, None)}

#: Field -> the CDC PLACES column whose per-ZIP crude prevalence is that field's rate.
PLACES_RATES = {
    "mobility_impaired": "mobility_crudeprev",
    "low_assets": "shututility_crudeprev",   # the measured "owns an AC, cannot run it"
    "transport_barrier": "lacktrpt_crudeprev",
    "no_caregiver": "emotionspt_crudeprev",  # lacks social and emotional support
    "copd": "copd_crudeprev",
    "asthma": "casthma_crudeprev",
    "active_cancer_tx": "cancer_crudeprev",
    "depression": "depression_crudeprev",
    "diabetes": "diabetes_crudeprev",
    # PLACES publishes coronary heart disease, not heart failure. CHD is the nearest
    # measured per-ZIP cardiac rate, so CHF is proxied by it. Say so if asked.
    "chf": "chd_crudeprev",
    # SNAP receipt, a measured per-ZIP floor on the share with low income.
    "low_income": "foodstamp_crudeprev",
}

#: Past-year PTSD by service era. VA National Center for PTSD, "How Common Is PTSD in
#: Veterans?", https://www.ptsd.va.gov/understand/common/common_veterans.asp :
#: OEF/OIF 15%, Gulf War 14%, Vietnam 5%. The page gives no peacetime figure, so peacetime
#: takes its lowest one (WWII/Korea, 2%). That last number is an assumption.
PTSD_PAST_YEAR = {"post911": 0.15, "gulf": 0.14, "vietnam": 0.05, "peacetime": 0.02}

#: Sites that can dispense methadone for an opioid treatment program (docs/SPEC.md §5.2).
OTP_STATIONS = ("630", "630A4")

# --------------------------------------------------------------------------- #
# ASSUMPTIONS. No public per-person or per-ZIP source exists for any of these. They are
# parametric, every column they feed is `_synthetic`, and any of them can be overruled
# here without touching the code below. "(SPEC)" means the value is from docs/SPEC.md §5.3.
# --------------------------------------------------------------------------- #

#: Age inside the open ACS 75+ band: 75 + Exponential(mean 7), so the band averages ~82.
OLDEST_BAND_MEAN_EXCESS = 7.0
MAX_AGE = 104

#: P(service era | age in 2026). Vietnam era 1964-75, Gulf 1990-2001, post-9/11 after.
#: WWII and Korea fold into "peacetime" because the contract has four eras.
ERA_BY_AGE = (
    (18, 42, {"post911": 1.0}),
    (43, 55, {"post911": 0.5, "gulf": 0.5}),
    (56, 68, {"gulf": 0.45, "peacetime": 0.55}),
    (69, 74, {"vietnam": 0.5, "peacetime": 0.5}),
    (75, 89, {"vietnam": 0.75, "peacetime": 0.25}),
    (90, MAX_AGE, {"peacetime": 1.0}),
)

#: Burn-pit years for gulf and post-9/11 veterans: LogNormal with mean 0.8 (SPEC).
BURN_PIT_MEAN_YEARS, BURN_PIT_SIGMA = 0.8, 0.75
#: pact_presumptive observes exposure noisily: true positive 0.6, false positive 0.05 (SPEC).
PACT_TRUE_POSITIVE, PACT_FALSE_POSITIVE = 0.60, 0.05
#: PTSD severity 1..4, given PTSD.
PTSD_SEVERITY = {1: 0.30, 2: 0.35, 3: 0.25, 4: 0.10}

#: emPOWER counts Medicare beneficiaries, who are overwhelmingly 65+. Under 65 the ZIP's
#: rate is scaled down rather than applied as-is.
UNDER_65_EQUIPMENT_RATIO = 0.25
#: A ZIP with 143 residents over 65 can put 39 of them on emPOWER. Cap the rate.
EQUIPMENT_RATE_CAP = 0.10
#: Which device, for powered equipment other than oxygen. emPOWER does not split per ZIP.
NON_OXYGEN_DEVICE = {"ventilator": 0.45, "wheelchair": 0.30, "bed": 0.25}
#: Share of dialysis patients who dialyse at home.
HOME_DIALYSIS_SHARE = 0.12
#: Enrolled in a methadone opioid treatment program (SPEC).
OTP_RATE = 0.01

LIVES_ALONE = {"under_65": 0.20, "65_plus": 0.30}
#: Caregiver type given one exists (SPEC). Living alone turns coresident into remote.
CAREGIVER_TYPE = {"informal_coresident": 0.55, "informal_remote": 0.35, "va_pcafc": 0.10}
CAREGIVER_CONTACT_CONSENT = 0.70
HIGH_INCOME_GIVEN_NOT_LOW = 0.40

#: Home AC falls with HVI band (the index is built partly from AC access) and with low
#: assets. HVI 1 -> 95%, HVI 5 -> 79%, minus 10 points for low assets.
HOME_AC_AT_HVI_1, HOME_AC_PER_HVI_BAND, HOME_AC_LOW_ASSETS = 0.95, 0.04, 0.10

#: Floor, citywide (SPEC). Basement doubles where more than 15% of the ZIP floods in
#: NYC's moderate-rain stormwater scenario.
FLOOR_P = {"basement": 0.06, "ground": 0.25, "upper": 0.69}
BASEMENT_FLOOD_MULTIPLIER, FLOOD_PRONE_FRAC = 2.0, 0.15

CONSENT = {"consent_partner_check": 0.60, "consent_ride": 0.70, "consent_housing": 0.50}

#: 12-month utilisation, Poisson means. "Other chronic" covers what the contract does
#: not name: hypertension, hyperlipidaemia, arthritis.
OTHER_CHRONIC = {"under_65": 1.0, "65_plus": 2.5}
ER_BASE, ER_PER_CONDITION = 0.20, 0.15
MISSED_REFILLS_BASE, MISSED_REFILLS_PER_CONDITION = 0.30, 0.10
MISSED_APPTS_BASE, MISSED_APPTS_TRANSPORT, MISSED_APPTS_DEPRESSION = 0.60, 0.60, 0.40

FIRST_M = ["Walter", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
           "Thomas", "Charles", "Anthony", "Raymond", "Frank", "Dennis", "Gerald", "Luis",
           "Jose", "Carlos", "Andre", "Marcus", "Darnell", "Kevin", "Brian", "Victor",
           "Hector", "Wei", "Min-jun", "Rajesh", "Samuel", "Eugene", "Harold", "Stanley"]
FIRST_F = ["Linda", "Patricia", "Carmen", "Gloria", "Denise", "Yolanda", "Marta", "Sharon",
           "Karen", "Angela", "Tanya", "Maria", "Keisha", "Lisa", "Mei", "Priya"]
LAST = ["Reyes", "Okafor", "Delgado", "Nguyen", "Brooks", "Castellano", "Whitfield",
        "Ferreira", "Kowalski", "Ramsey", "Baptiste", "Ortiz", "Coleman", "Vasquez",
        "Murphy", "Rosen", "Goldberg", "Washington", "Jackson", "Rivera", "Santiago",
        "Chen", "Wong", "Patel", "Singh", "Russo", "Esposito", "O'Brien", "Kelly",
        "Johnson", "Williams", "Davis", "Morales", "Cruz", "Haddad", "Petrov", "Kim",
        "Lee", "Harris", "Greene"]


# --------------------------------------------------------------------------- #
# Randomness: one named stream per component
# --------------------------------------------------------------------------- #

def _stream(seed: int, name: str) -> np.random.Generator:
    """An independent random stream per component, keyed by name.

    Adding a column later draws from a new stream, so it cannot silently reshuffle where
    everyone lives or who has COPD. The same click gives the same number on stage.
    """
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def _categorical(u: np.ndarray, probs: dict) -> np.ndarray:
    """Map uniforms onto the keys of `probs` by inverse CDF."""
    labels = np.array(list(probs))
    cdf = np.cumsum(list(probs.values()))
    cdf = cdf / cdf[-1]
    return labels[np.minimum(np.searchsorted(cdf, u, side="right"), len(labels) - 1)]


# --------------------------------------------------------------------------- #
# The ZIP frame: every per-ZIP rate and exposure, one row per MODZCTA
# --------------------------------------------------------------------------- #

def _ref(name: str) -> pl.DataFrame:
    return pl.read_parquet(REFERENCE / f"{name}.parquet")


def _members() -> pl.DataFrame:
    """MODZCTA -> member ZCTA. ACS and emPOWER are published per ZCTA/ZIP, not MODZCTA."""
    return (_ref("nyc_modzcta")
            .select("modzcta", pl.col("zcta_members").str.split(",").alias("zcta"))
            .explode("zcta", empty_as_null=True).with_columns(pl.col("zcta").str.strip_chars()))


def _per_65plus(count: str, alias: str) -> pl.Expr:
    return (pl.when(pl.col("pop_65plus") > 0)
              .then(pl.col(count) / pl.col("pop_65plus")).otherwise(0.0)
              .clip(0.0, EQUIPMENT_RATE_CAP).alias(alias))


def zip_frame() -> pl.DataFrame:
    """One row per MODZCTA that has every source: counts to sample from, rates to draw at."""
    members = _members()
    acs_cols = [f"vet_{s}_{b}" for s in ("m", "f") for b in BANDS] + ["pop_65plus"]
    acs = (_ref("acs_veterans_by_zcta").join(members, on="zcta")
           .group_by("modzcta").agg(pl.col(acs_cols).sum()))
    empower = (_ref("empower_ny_zip").join(members, left_on="zip", right_on="zcta")
               .group_by("modzcta")
               .agg(pl.col("dme_power_dependent", "dme_oxygen", "dme_esrd_dialysis").sum()))
    borough = _ref("empower_ny_zip").select(pl.col("zip").alias("modzcta"), "borough")
    places = _ref("places_zcta_nyc").select(
        pl.col("zcta").alias("modzcta"),
        *[(pl.col(col) / 100).alias(f"rate_{field}") for field, col in PLACES_RATES.items()])
    hvi = _ref("hvi_by_zcta").select(pl.col("zcta").alias("modzcta"),
                                     pl.col("hvi").cast(pl.Int32))
    evac = _ref("evac_zone_by_modzcta").select(
        "modzcta", pl.col("evac_zone_min").cast(pl.Int32).alias("evac_zone"))
    storm = _ref("stormwater_by_modzcta")

    frame = _ref("nyc_modzcta").select("modzcta", "lon", "lat")
    for table in (acs, empower, borough, places, hvi, evac, storm):
        frame = frame.join(table, on="modzcta", how="inner")
    frame = frame.filter(pl.col("borough").is_in(schema.BOROUGHS))
    return (frame.with_columns(
                _per_65plus("dme_power_dependent", "rate_powered"),
                _per_65plus("dme_oxygen", "rate_oxygen"),
                _per_65plus("dme_esrd_dialysis", "rate_dialysis"))
            .with_columns(pl.min_horizontal("rate_oxygen", "rate_powered").alias("rate_oxygen"))
            .sort("modzcta"))


# --------------------------------------------------------------------------- #
# Re-homing: where each veteran lives, their age and their sex
# --------------------------------------------------------------------------- #

def rehome(n: int, seed: int, zips: pl.DataFrame) -> pl.DataFrame:
    """Sample (modzcta, sex, age band) jointly in proportion to ACS veteran counts."""
    cells = (zips.select("modzcta", *[f"vet_{s}_{b}" for s in ("m", "f") for b in BANDS])
             .unpivot(index="modzcta", variable_name="cell", value_name="veterans")
             .with_columns(pl.col("cell").str.slice(4, 1).str.to_uppercase().alias("sex"),
                           pl.col("cell").str.slice(6).alias("band"))
             .filter(pl.col("veterans") > 0)
             .sort("modzcta", "sex", "band"))

    acs_total = _ref("acs_veterans_by_zcta")["veterans_total"].sum()
    covered = cells["veterans"].sum() / acs_total
    if covered < 0.99:
        raise SystemExit(f"only {covered:.1%} of ACS veterans live in ZIPs with every source; "
                         "a reference join is broken")

    w = cells["veterans"].to_numpy()
    pick = _stream(seed, "rehome").choice(cells.height, size=n, p=w / w.sum())
    people = cells[pick].select("modzcta", "sex", "band")

    r = _stream(seed, "age")
    u, tail = r.random(n), r.exponential(OLDEST_BAND_MEAN_EXCESS, n)
    band = people["band"].to_numpy()
    lo = np.array([BANDS[b][0] for b in band])
    hi = np.array([BANDS[b][1] or MAX_AGE for b in band])
    age = np.where(band == "75plus",
                   np.minimum(75 + np.floor(tail), MAX_AGE),
                   lo + np.floor(u * (hi - lo + 1)))
    return (people.with_columns(pl.Series("age", age.astype(np.int32)))
                  .with_row_index("_row")
                  .join(zips, on="modzcta", how="left")
                  .sort("_row").drop("_row"))


# --------------------------------------------------------------------------- #
# Facility assignment
# --------------------------------------------------------------------------- #

def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def assign_facility(people: pl.DataFrame, dialysis: np.ndarray, otp: np.ndarray) -> np.ndarray:
    fac = _ref("va_facilities_nyc_hazard")
    # Vet Centers counsel rather than treat, and the mobile clinic has no fixed site.
    care = (fac.filter(~pl.col("station_no").str.ends_with("V")
                       & ~pl.col("name").str.contains("Mobile"))
               .sort("station_no"))
    station = care["station_no"].to_numpy()
    dist = _haversine_km(people["lat"].to_numpy()[:, None], people["lon"].to_numpy()[:, None],
                         care["lat"].to_numpy()[None, :], care["lon"].to_numpy()[None, :])

    def nearest(allowed: np.ndarray) -> np.ndarray:
        return station[np.where(allowed, dist, np.inf).argmin(axis=1)]

    same_borough = people["borough"].to_numpy()[:, None] == care["borough"].to_numpy()[None, :]
    site_dependent = np.broadcast_to(care["site_dependent_services"].to_numpy(), dist.shape)
    otp_site = np.broadcast_to(np.isin(station, OTP_STATIONS), dist.shape)

    out = nearest(same_borough)
    out = np.where(dialysis, nearest(site_dependent), out)
    return np.where(otp, nearest(otp_site), out)


# --------------------------------------------------------------------------- #
# Augment: everything about the person beyond where they live
# --------------------------------------------------------------------------- #

def augment(people: pl.DataFrame, seed: int) -> pl.DataFrame:
    n = people.height
    col = lambda name: people[name].to_numpy()  # noqa: E731

    def bern(name: str, p) -> np.ndarray:
        return _stream(seed, name).random(n) < p

    def uniform(name: str) -> np.ndarray:
        return _stream(seed, name).random(n)

    age, sex = col("age"), col("sex")
    older = age >= 65

    # CDC PLACES: each veteran is one Bernoulli draw at their own ZIP's measured rate.
    places = {field: bern(field, col(f"rate_{field}")) for field in PLACES_RATES}

    # Service era, burn pits, PACT and PTSD.
    era = np.empty(n, dtype=object)
    u_era = uniform("era")
    for lo, hi, probs in ERA_BY_AGE:
        m = (age >= lo) & (age <= hi)
        era[m] = _categorical(u_era[m], probs)
    era = era.astype(str)
    exposed = np.isin(era, ["gulf", "post911"])
    mu = np.log(BURN_PIT_MEAN_YEARS) - BURN_PIT_SIGMA ** 2 / 2
    burn = np.where(exposed, _stream(seed, "burn_pit").lognormal(mu, BURN_PIT_SIGMA, n), 0.0)
    pact = bern("pact", np.where(exposed, PACT_TRUE_POSITIVE, PACT_FALSE_POSITIVE))
    ptsd = bern("ptsd", np.vectorize(PTSD_PAST_YEAR.get)(era))
    severity = np.where(ptsd, _categorical(uniform("ptsd_severity"), PTSD_SEVERITY), 0)

    # Dialysis and powered equipment, at the ZIP's emPOWER rate.
    scale = np.where(older, 1.0, UNDER_65_EQUIPMENT_RATIO)
    dialysis = bern("dialysis", col("rate_dialysis") * scale)
    u_eq = uniform("equipment")
    device = _categorical(uniform("device"), NON_OXYGEN_DEVICE)
    equipment = np.where(u_eq < col("rate_oxygen") * scale, "oxygen",
                         np.where(u_eq < col("rate_powered") * scale, device, "none"))
    equipment = np.where(dialysis & bern("home_dialysis", HOME_DIALYSIS_SHARE),
                         "dialysis_home", equipment)
    otp = bern("otp", OTP_RATE)

    # Household, caregiver and money.
    lives_alone = bern("lives_alone",
                       np.where(older, LIVES_ALONE["65_plus"], LIVES_ALONE["under_65"]))
    kind = _categorical(uniform("caregiver_type"), CAREGIVER_TYPE)
    kind = np.where(lives_alone & (kind == "informal_coresident"), "informal_remote", kind)
    caregiver = np.where(places["no_caregiver"], "none", kind)
    caregiver_consent = (caregiver != "none") & bern("caregiver_consent",
                                                     CAREGIVER_CONTACT_CONSENT)

    # Low assets implies low income. The remaining low-income draw is sized so the ZIP's
    # low-income share still equals its SNAP rate: s + (1 - s) q = f.
    s, f = col("rate_low_assets"), col("rate_low_income")
    q = np.clip((f - s) / np.maximum(1 - s, 1e-9), 0, 1)
    low_income = places["low_assets"] | bern("low_income_rest", q)
    income = np.where(low_income, "low",
                      np.where(bern("high_income", HIGH_INCOME_GIVEN_NOT_LOW), "high", "mid"))

    # Housing.
    hvi = col("hvi")
    home_ac = bern("home_ac", HOME_AC_AT_HVI_1 - HOME_AC_PER_HVI_BAND * (hvi - 1)
                   - HOME_AC_LOW_ASSETS * places["low_assets"])
    flood_prone = col("stormwater_flooded_frac") > FLOOD_PRONE_FRAC
    p_base = FLOOR_P["basement"] * np.where(flood_prone, BASEMENT_FLOOD_MULTIPLIER, 1.0)
    total = p_base + FLOOR_P["ground"] + FLOOR_P["upper"]
    u_floor = uniform("floor")
    floor = np.where(u_floor < p_base / total, "basement",
                     np.where(u_floor < (p_base + FLOOR_P["ground"]) / total, "ground", "upper"))

    # Chronic load and 12-month utilisation.
    named = sum(x.astype(int) for x in (
        places["copd"], places["asthma"], places["chf"], places["diabetes"], dialysis,
        places["active_cancer_tx"], ptsd, places["depression"]))
    other = _stream(seed, "other_chronic").poisson(
        np.where(older, OTHER_CHRONIC["65_plus"], OTHER_CHRONIC["under_65"]))
    n_chronic = np.clip(named + other, 0, 30)
    er = _stream(seed, "er").poisson(ER_BASE + ER_PER_CONDITION * n_chronic)
    missed_refills = _stream(seed, "missed_refills").poisson(
        MISSED_REFILLS_BASE + MISSED_REFILLS_PER_CONDITION * n_chronic)
    missed_appts = _stream(seed, "missed_appts").poisson(
        MISSED_APPTS_BASE + MISSED_APPTS_TRANSPORT * places["transport_barrier"]
        + MISSED_APPTS_DEPRESSION * places["depression"])

    r = _stream(seed, "names")
    first_m, first_f = r.integers(len(FIRST_M), size=n), r.integers(len(FIRST_F), size=n)
    last = r.integers(len(LAST), size=n)
    names = [f"{FIRST_F[a] if s_ == 'F' else FIRST_M[b]} {LAST[c]}"
             for s_, a, b, c in zip(sex, first_f, first_m, last, strict=True)]

    i32 = lambda x: pl.Series(np.asarray(x).astype(np.int32))  # noqa: E731
    df = pl.DataFrame({
        "veteran_id": [f"SYN-{i:06d}" for i in range(n)],
        "name_display": names,
        "age": i32(age),
        "sex": sex,
        "race": pl.Series([None] * n, dtype=pl.Utf8),
        "ethnicity": pl.Series([None] * n, dtype=pl.Utf8),
        "modzcta": col("modzcta"),
        "borough": col("borough"),
        "facility_id": assign_facility(people, dialysis, otp),
        "copd": places["copd"],
        "asthma": places["asthma"],
        "chf": places["chf"],
        "diabetes": places["diabetes"],
        "ckd_dialysis": dialysis,
        "active_cancer_tx": places["active_cancer_tx"],
        "ptsd": ptsd,
        "depression": places["depression"],
        "pact_presumptive": pact,
        "n_chronic": i32(n_chronic),
        "er_visits_12m": i32(np.clip(er, 0, 100)),
        "missed_refills_12m": i32(np.clip(missed_refills, 0, 100)),
        "missed_appts_12m": i32(np.clip(missed_appts, 0, 100)),
        "deployment_era": era,
        "burn_pit_years": np.clip(burn, 0, 20).round(2),
        "ptsd_severity": i32(severity),
        "home_ac": home_ac,
        "floor": floor,
        "on_methadone_otp": otp,
        "powered_equipment": equipment,
        "lives_alone": lives_alone,
        "mobility_impaired": places["mobility_impaired"],
        "caregiver": caregiver,
        "caregiver_contact_consent": caregiver_consent,
        "income_band": income,
        "low_assets": places["low_assets"],
        "transport_barrier": places["transport_barrier"],
        "evac_zone": i32(col("evac_zone")),
        "stormwater_flooded_frac": col("stormwater_flooded_frac"),
        "hvi": i32(hvi),
        **{name: bern(name, p) for name, p in CONSENT.items()},
    })
    # Prescriptions, the mechanisms they carry, and pharmacy logistics (medications.py).
    df = medications.attach(df, seed)

    contract = schema.TABLES["cohort"].columns
    flags = [pl.lit(True).alias(f"{c.name}_synthetic") for c in contract if c.synthetic]
    order = [name for c in contract
             for name in ((c.name, f"{c.name}_synthetic") if c.synthetic else (c.name,))]
    return df.with_columns(flags).select(order)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def build(n: int = N_DEFAULT, seed: int = 0) -> pl.DataFrame:
    """The cohort as a validated frame. Pure: reads data/reference/, writes nothing."""
    return schema.validate(augment(rehome(n, seed, zip_frame()), seed), "cohort")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="veterans to generate")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = build(args.n, args.seed)
    path = schema.write(df, "cohort")
    share = lambda e: f"{df.select(e.mean()).item():.1%}"  # noqa: E731
    print(f"  {df.height:,} veterans in {df['modzcta'].n_unique()} ZIPs -> {path}")
    print(f"  65+ {share(pl.col('age') >= 65)} · female {share(pl.col('sex') == 'F')} · "
          f"mobility {share(pl.col('mobility_impaired'))} · low assets "
          f"{share(pl.col('low_assets'))} · no caregiver {share(pl.col('caregiver') == 'none')}")
    print(f"  powered equipment {share(pl.col('powered_equipment') != 'none')} · dialysis "
          f"{df['ckd_dialysis'].sum()} · OTP {df['on_methadone_otp'].sum()} "
          f"({df.filter(pl.col('on_methadone_otp') & (pl.col('facility_id') == '630')).height}"
          f" at station 630)")
    print(f"  meds: {df['n_active_meds'].mean():.1f} active each · heat-impairing "
          f"{share(pl.col('med_thermoreg_score') > 0)} · CDC pair "
          f"{share(pl.col('med_combo_raas_diuretic'))} · cold chain "
          f"{share(pl.col('med_cold_chain'))} · controlled "
          f"{share(pl.col('med_controlled'))} -> {df['med_controlled'].sum()} for the pharmacist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
