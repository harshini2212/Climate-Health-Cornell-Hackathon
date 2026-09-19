"""Acceptance tests for `leeward/cohort/medications.py` -- the medication layer.

Written before the module. The claim they protect is the one said on stage: *every
medication flag on a veteran card traces to a real RxNorm code, the VA's own drug class,
and CDC's published mechanism list.* So nothing here hand-writes a drug list. The tests
read the same two committed crosswalks the module reads, and every prevalence they assert
was **measured** on Synthea's public FHIR sample rather than chosen -- see
`test_the_committed_profiles_are_still_the_synthea_sample`, which re-derives the profile
table from the 30 MB bundle zip whenever `data/raw/` is present.

What is legitimately synthetic -- which real medication list a synthetic veteran carries,
and their pharmacy logistics -- is asserted to carry a `_synthetic` flag.
"""

from __future__ import annotations

import polars as pl
import pytest

from leeward import schema
from leeward.cohort import build, medications

REF = schema.REFERENCE

# --------------------------------------------------------------------------- #
# Anchors. Real RxNorm codes, each verified against the committed crosswalk by
# test_the_anchor_codes_are_really_in_the_committed_crosswalk below -- so if a future
# RxNav pull moves them, one test names the problem instead of nine failing obscurely.
#
# The first two are the pair CDC singles out, and both are top-five drugs in the Synthea
# sample: these terms fire on the ordinary patient, not an exotic one.
# --------------------------------------------------------------------------- #
HCTZ = "310798"         # hydrochlorothiazide 25 MG -> CV701 thiazide (and CV700, its parent)
LISINOPRIL = "314076"   # lisinopril 10 MG          -> CV800 ACE inhibitor
LOSARTAN = "979492"     # losartan 50 MG            -> CV805 angiotensin II inhibitor
FUROSEMIDE = "313988"   # furosemide 40 MG          -> CV702 loop diuretic (and CV700)
IBUPROFEN = "197805"    # ibuprofen 400 MG          -> MS102 NSAID
INSULIN = "106892"      # Humulin 70/30             -> HS501 insulin (and HS500)
ALPRAZOLAM = "308047"   # alprazolam 0.25 MG        -> CN302 benzodiazepine (and CN300)
OXYCODONE = "1049221"   # oxycodone/APAP 5/325      -> CN101 opioid analgesic
WARFARIN = "855332"     # warfarin sodium 5 MG      -> BL110 anticoagulant
AMLODIPINE = "197361"   # amlodipine 5 MG           -> CV200 calcium channel blocker
NOT_A_DRUG = "999999999"

#: Prevalence among the Synthea patients who have at least one *active* medication
#: (77 of the 109 bundles). Measured, not chosen. `any_climate_class` is the number
#: `data/README.md` used to print as 77% under the heat heading -- it is the share on any
#: medication in the crosswalk, of any hazard, not the share on a heat-mechanism drug.
SYNTHEA_PREVALENCE = {
    "any_climate_class": 0.766,
    "med_thermoreg_score": 0.649,   # > 0
    "med_combo_raas_diuretic": 0.156,
    "med_controlled": 0.104,
    "med_cold_chain": 0.091,
    "med_narrow_ti": 0.078,
    "acb_score": 0.117,             # >= 3, the clinical threshold
    "med_renal_triple": 0.013,
}
TOLERANCE = 0.10  # percentage points, per docs/PROMPTS.md


def _row(*rxcuis: str) -> dict:
    """The derived medication block for one veteran holding these active codes."""
    return medications.derive([list(rxcuis)]).row(0, named=True)


def _weight(va_class: str) -> float:
    risk = pl.read_csv(REF / "med_climate_risk.csv")
    return risk.filter(pl.col("va_class_id") == va_class)["weight"].item()


# --------------------------------------------------------------------------- #
# The crosswalks themselves. If these rot, everything below is meaningless.
# --------------------------------------------------------------------------- #

def test_the_anchor_codes_are_really_in_the_committed_crosswalk() -> None:
    xw = pl.read_parquet(REF / "va_drug_class_members.parquet")
    known = set(xw["rxcui"])
    anchors = {HCTZ, LISINOPRIL, LOSARTAN, FUROSEMIDE, IBUPROFEN, INSULIN,
               ALPRAZOLAM, OXYCODONE, WARFARIN, AMLODIPINE}
    assert anchors <= known, f"RxNav pull no longer carries: {sorted(anchors - known)}"
    assert NOT_A_DRUG not in known


def test_the_layer_reads_the_committed_crosswalks_and_reaches_no_network() -> None:
    """The crosswalks are committed precisely so RxNav is not a demo-day dependency."""
    src = (schema.ROOT / "leeward" / "cohort" / "medications.py").read_text()
    assert "med_climate_risk.csv" in src and "va_drug_class_members.parquet" in src
    assert "http" not in src.lower(), "the medication layer must not carry a URL"
    for module in ("requests", "httpx", "urllib"):
        assert f"import {module}" not in src, f"{module} has no business in the demo path"


# --------------------------------------------------------------------------- #
# Derivation. docs/SPEC.md 5.3 says "sum over classes"; a drug sits in both its VA class
# and that class's parent, so the sum is taken over the *most specific* class per drug.
# --------------------------------------------------------------------------- #

def test_a_drug_resolves_to_its_most_specific_va_class() -> None:
    """A pharmacist says "thiazide", not "diuretic"; "insulin", not "glucose regulation
    agent". Counting both would double every parented drug's weight and ACB."""
    assert medications.classes_for([HCTZ]) == ["CV701"]
    assert medications.classes_for([FUROSEMIDE]) == ["CV702"]
    assert medications.classes_for([INSULIN]) == ["HS501"]
    assert medications.classes_for([ALPRAZOLAM]) == ["CN302"]
    assert medications.classes_for([LISINOPRIL]) == ["CV800"]


def test_two_drugs_give_the_union_of_their_classes() -> None:
    assert medications.classes_for([HCTZ, LISINOPRIL]) == ["CV701", "CV800"]


def test_the_cdc_named_pair_scores_the_sum_of_its_two_mechanisms() -> None:
    """Hydrochlorothiazide plus lisinopril: volume depletion plus blunted thirst. The
    weights come out of med_climate_risk.csv, so a pharmacist editing that file moves
    this number and no code changes."""
    r = _row(HCTZ, LISINOPRIL)
    assert r["med_combo_raas_diuretic"] is True
    assert r["med_thermoreg_score"] == pytest.approx(_weight("CV701") + _weight("CV800"))
    assert r["n_active_meds"] == 2


def test_an_arb_also_counts_as_the_raas_half() -> None:
    assert _row(FUROSEMIDE, LOSARTAN)["med_combo_raas_diuretic"] is True


def test_a_diuretic_on_its_own_is_not_the_combination() -> None:
    assert _row(HCTZ)["med_combo_raas_diuretic"] is False
    assert _row(LISINOPRIL)["med_combo_raas_diuretic"] is False


def test_renal_triple_needs_the_nsaid_on_top_of_both() -> None:
    assert _row(HCTZ, LISINOPRIL, IBUPROFEN)["med_renal_triple"] is True
    assert _row(HCTZ, LISINOPRIL)["med_renal_triple"] is False
    assert _row(IBUPROFEN)["med_renal_triple"] is False


def test_insulin_is_cold_chain_and_carries_no_heat_mechanism() -> None:
    """Insulin's hazard is the outage, not the heat -- it spoils in about a day."""
    r = _row(INSULIN)
    assert r["med_cold_chain"] is True
    assert r["med_thermoreg_score"] == 0.0


def test_a_benzodiazepine_is_controlled_so_retail_refill_cannot_cover_it() -> None:
    """The VA disaster pharmacy benefit excludes controlled substances, which is what
    makes these veterans need an earlier and different action."""
    assert _row(ALPRAZOLAM)["med_controlled"] is True
    assert _row(OXYCODONE)["med_controlled"] is True
    assert _row(AMLODIPINE)["med_controlled"] is False


def test_narrow_therapeutic_index_is_flagged() -> None:
    assert _row(WARFARIN)["med_narrow_ti"] is True
    assert _row(AMLODIPINE)["med_narrow_ti"] is False


def test_acb_sums_the_scale_over_classes_not_over_drugs() -> None:
    """ACB is a 0-3 burden per mechanism. Two drugs in one class is one mechanism."""
    one = _row(ALPRAZOLAM)["acb_score"]
    assert 0 <= one <= 3
    assert _row(ALPRAZOLAM, ALPRAZOLAM)["acb_score"] == one
    assert _row(ALPRAZOLAM, WARFARIN)["acb_score"] == one  # BL110 has no ACB


def test_no_medications_means_no_flags() -> None:
    r = _row()
    assert r["n_active_meds"] == 0
    assert r["va_drug_classes"] == []
    assert r["med_thermoreg_score"] == 0.0
    assert r["acb_score"] == 0
    for flag in ("med_combo_raas_diuretic", "med_renal_triple", "med_cold_chain",
                 "med_controlled", "med_narrow_ti"):
        assert r[flag] is False


def test_a_code_outside_the_crosswalk_is_counted_but_carries_no_mechanism() -> None:
    """Statins, vitamins and contraceptives are real active meds with no climate
    mechanism. They must not crash the join and must not invent a flag."""
    r = _row(NOT_A_DRUG, HCTZ)
    assert r["n_active_meds"] == 2
    assert r["va_drug_classes"] == ["CV701"]
    assert r["med_thermoreg_score"] == pytest.approx(_weight("CV701"))


def test_derive_is_row_wise_and_keeps_the_order_it_was_given() -> None:
    out = medications.derive([[INSULIN], [], [ALPRAZOLAM]])
    assert out.height == 3
    assert out["med_cold_chain"].to_list() == [True, False, False]
    assert out["med_controlled"].to_list() == [False, False, True]


# --------------------------------------------------------------------------- #
# The Synthea sample. `data/reference/synthea_med_profiles.parquet` is the distilled
# form: one row per bundle, the patient's active RxNorm codes. It is committed so a
# clean clone has real medication lists without the 30 MB of FHIR.
# --------------------------------------------------------------------------- #

def test_the_profile_table_is_one_row_per_synthea_bundle() -> None:
    p = medications.load_profiles()
    assert p.height == 109, "the published sample is 109 bundles"
    assert p["profile_id"].n_unique() == p.height
    assert p["age"].min() >= 0 and p["age"].max() < 120
    assert set(p["sex"].unique()) <= {"M", "F"}
    assert p["rxcuis"].list.len().sum() > 0


def test_prevalence_matches_what_was_measured_on_the_synthea_sample() -> None:
    """The acceptance numbers in docs/PROMPTS.md, each within 10 percentage points.

    Denominator: the 77 bundles with at least one active medication. A patient on no
    medication is not evidence about medication risk.
    """
    p = medications.load_profiles().filter(pl.col("rxcuis").list.len() > 0)
    assert p.height == 77
    d = medications.derive(p["rxcuis"].to_list())

    got = {
        "any_climate_class": d["va_drug_classes"].list.len().gt(0).mean(),
        "med_thermoreg_score": d["med_thermoreg_score"].gt(0).mean(),
        "med_combo_raas_diuretic": d["med_combo_raas_diuretic"].mean(),
        "med_controlled": d["med_controlled"].mean(),
        "med_cold_chain": d["med_cold_chain"].mean(),
        "med_narrow_ti": d["med_narrow_ti"].mean(),
        "acb_score": d["acb_score"].ge(3).mean(),
        "med_renal_triple": d["med_renal_triple"].mean(),
    }
    off = {k: (round(got[k], 3), v) for k, v in SYNTHEA_PREVALENCE.items()
           if abs(got[k] - v) > TOLERANCE}
    assert not off, f"prevalence drifted from the sample (got, expected): {off}"


def test_the_median_patient_is_on_three_medications() -> None:
    p = medications.load_profiles().filter(pl.col("rxcuis").list.len() > 0)
    n = p["rxcuis"].list.len()
    assert n.median() == 3
    assert n.max() == 14


def test_medication_burden_rises_with_age_in_the_sample() -> None:
    """Why the cohort stratifies its draw by age band at all: a 55+ Synthea patient
    carries roughly three times the medication list of an 18-54 one."""
    p = medications.load_profiles().filter(pl.col("age") >= 18)
    young = p.filter(pl.col("age") < medications.OLDER_PROFILE_AGE)["rxcuis"].list.len().mean()
    old = p.filter(pl.col("age") >= medications.OLDER_PROFILE_AGE)["rxcuis"].list.len().mean()
    assert old > 2 * young, f"expected a strong age gradient, got {young:.1f} vs {old:.1f}"


def test_the_committed_profiles_are_still_the_synthea_sample() -> None:
    """Re-derive from the bundles themselves. Skips on a clean clone, where data/raw/ is
    absent by design -- this is the check that the committed table was not edited."""
    src = schema.DATA / "raw" / "synthea_sample_fhir.zip"
    if not src.exists():
        pytest.skip("data/raw/synthea_sample_fhir.zip absent -- run `make sources-heavy`")
    fresh = medications.profiles_from_fhir(src)
    committed = medications.load_profiles()
    assert fresh.sort("profile_id").equals(committed.sort("profile_id"))


# --------------------------------------------------------------------------- #
# Attaching the layer to the cohort.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def cohort() -> pl.DataFrame:
    return build.build(n=2000, seed=0)


def test_the_cohort_still_satisfies_its_contract(cohort: pl.DataFrame) -> None:
    schema.validate(cohort, "cohort")


def test_every_medication_column_is_filled_not_zeroed(cohort: pl.DataFrame) -> None:
    """Before this module landed, all seven were zero for all 10,000 -- which meant
    pharmacist_med_review and cold_chain_plan could never fire."""
    assert cohort["n_active_meds"].sum() > 0
    assert cohort["med_thermoreg_score"].gt(0).mean() > 0.5
    for flag in ("med_combo_raas_diuretic", "med_cold_chain", "med_controlled",
                 "med_narrow_ti"):
        assert cohort[flag].sum() > 0, f"{flag} is still zero for everyone"


def test_every_veterans_medication_list_is_a_real_one(cohort: pl.DataFrame) -> None:
    """Synthetic people, real prescriptions. Every list in the cohort is some Synthea
    patient's actual active list -- none was assembled drug by drug."""
    real = {tuple(r) for r in medications.load_profiles()["rxcuis"].to_list()}
    got = {tuple(r) for r in cohort["med_rxcuis"].to_list()}
    assert got <= real, "a medication list in the cohort is not one the sample contains"


def test_the_derived_columns_agree_with_the_codes_they_came_from(cohort: pl.DataFrame
                                                                 ) -> None:
    """A wrong join in attach() cannot also fix its own test: re-derive from med_rxcuis."""
    again = medications.derive(cohort["med_rxcuis"].to_list())
    for col in again.columns:
        assert cohort[col].to_list() == again[col].to_list(), f"{col} does not match the codes"


def test_older_veterans_carry_the_heavier_medication_burden(cohort: pl.DataFrame) -> None:
    """data/README.md: "the veteran 65+ cohort will run higher". It has to be true here."""
    older = cohort.filter(pl.col("age") >= 65)
    younger = cohort.filter(pl.col("age") < medications.OLDER_PROFILE_AGE)
    assert older["med_thermoreg_score"].mean() > younger["med_thermoreg_score"].mean()
    assert older["n_active_meds"].mean() > younger["n_active_meds"].mean()


def test_no_veteran_is_prescribed_a_childs_medication_list(cohort: pl.DataFrame) -> None:
    """The cohort is 18+; the published sample is not. Paediatric bundles are excluded
    from the draw, so no 78-year-old inherits a 6-year-old's prescriptions."""
    adult = medications.load_profiles().filter(pl.col("age") >= 18)
    assert {tuple(r) for r in cohort["med_rxcuis"].to_list()} <= {
        tuple(r) for r in adult["rxcuis"].to_list()}


def test_mail_order_and_days_of_supply_are_drawn_at_the_published_va_rate(
        cohort: pl.DataFrame) -> None:
    """~80% of VA outpatient prescriptions go by mail (CMOP). 90-day mail fills, 30-day
    window fills, and the phase is uniform so the panel is spread across its refill cycle."""
    assert cohort["mail_order_pharmacy"].mean() == pytest.approx(
        medications.MAIL_ORDER_RATE, abs=0.03)
    mail = cohort.filter(pl.col("mail_order_pharmacy"))
    window = cohort.filter(~pl.col("mail_order_pharmacy"))
    assert mail["days_supply_remaining"].max() <= medications.MAIL_FILL_DAYS
    assert window["days_supply_remaining"].max() <= medications.WINDOW_FILL_DAYS
    assert cohort["days_supply_remaining"].min() >= 0
    # Uniform phase: the mean sits near the middle of the fill, not at an end.
    assert mail["days_supply_remaining"].mean() == pytest.approx(
        medications.MAIL_FILL_DAYS / 2, abs=6)


def test_what_is_synthetic_here_says_so(cohort: pl.DataFrame) -> None:
    """Which real list a synthetic veteran carries, and their pharmacy logistics, are
    the synthetic parts. The codes, classes and mechanisms are not."""
    for col in ("mail_order_pharmacy", "days_supply_remaining"):
        assert cohort[f"{col}_synthetic"].all()


def test_same_seed_same_medications_different_seed_different_ones() -> None:
    a = build.build(n=300, seed=0)
    b = build.build(n=300, seed=0)
    c = build.build(n=300, seed=1)
    assert a["med_rxcuis"].to_list() == b["med_rxcuis"].to_list()
    assert a["med_rxcuis"].to_list() != c["med_rxcuis"].to_list()
