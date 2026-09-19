"""The medication layer: RxNorm codes -> VA drug classes -> CDC's climate mechanisms.

    python -m leeward.cohort.medications            # what the layer does to the cohort

Synthea puts an RxNorm code on every prescription, so this block is **derived, not
invented**. Two committed crosswalks do all the work and neither needs the network:

    data/reference/va_drug_class_members.parquet    4,222 RxNorm codes -> VA drug class,
                                                    pulled once from RxNav
    data/reference/med_climate_risk.csv             52 VA classes -> mechanism, hazard,
                                                    weight, ACB, controlled / cold-chain /
                                                    narrow-TI, curated from CDC's clinician
                                                    guidance on heat and medications

RxNav publishes the VA's own 576-class taxonomy, which means Leeward speaks the vocabulary
a VA clinical pharmacist already uses: `CV702` is LOOP DIURETICS to RxNav, to the VA
formulary and to us. `med_climate_risk.csv` is pharmacist-editable -- change a weight there
and every score moves, with no code touched.

**Most specific class wins.** A drug sits in both its VA class and that class's parent:
hydrochlorothiazide is CV701 THIAZIDES *and* CV700 DIURETICS, insulin is HS501 *and* HS500.
docs/SPEC.md 5.3 says to sum over classes, which would count the same mechanism twice for
every parented drug -- doubling its weight and its ACB. So each drug contributes only its
most specific class. The hierarchy is not hand-written: it is read off the crosswalk, where
a child class's members are a strict subset of its parent's (27 such pairs, all clean).

**Leeward never changes a medication.** Everything here flags a veteran for the VA clinical
pharmacist, who decides. That is why `pharmacist_slot` is a scarce capacity unit in
`decision/allocate.py`: it is a real person's afternoon.

What is synthetic and flagged as such: `mail_order_pharmacy` and `days_supply_remaining`,
because Synthea's FHIR export carries no `dispenseRequest` block -- plus *which* real
medication list a synthetic veteran carries.

    Prescriptions are bootstrapped from `data/reference/synthea_med_profiles.parquet`, one
    row per Synthea bundle holding that patient's active RxNorm codes. Every list in the
    cohort is therefore some real (synthetic-but-generated-by-a-clinical-model) patient's
    actual list, co-prescribing intact, never assembled drug by drug. The draw is
    stratified by age band only, because the sample's medication burden triples at 55+
    (1.9 active meds under 55, 5.4 at 55+) and our panel is mostly older.

    **Known gap:** the draw ignores the veteran's own diagnosis list, so a cold-chain
    medication does not imply the diabetes flag. The sample has six diabetics -- too few to
    condition on without inventing the structure. This closes when the Synthea swap lands
    (docs/BUILD_PLAN.md, T+2:30), where one patient brings diagnoses and prescriptions
    together.
"""

from __future__ import annotations

import json
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl

from leeward.schema import REFERENCE

#: The CDC-named combination: an ACE inhibitor or an ARB, plus any diuretic.
RAAS_CLASSES = ("CV800", "CV805")
DIURETIC_CLASSES = ("CV700", "CV701", "CV702", "CV703", "CV704", "CV709")
#: On top of both, an NSAID -- acute kidney injury when the veteran is also dehydrated.
NSAID_CLASSES = ("MS101", "MS102")

#: ~80% of VA outpatient prescriptions go by mail through CMOP (docs/sources.md). Mail
#: fills are 90 days, window fills 30, and the phase is uniform, so on any given day the
#: panel is spread across its refill cycle.
MAIL_ORDER_RATE, MAIL_FILL_DAYS, WINDOW_FILL_DAYS = 0.80, 90, 30

#: The cohort is 18+; the published Synthea sample is not, so paediatric bundles are never
#: drawn. 55 is the band cut: it is where the sample's medication burden jumps, and it
#: keeps 23 profiles in the older pool where a 65 cut would leave 10.
ADULT_AGE, OLDER_PROFILE_AGE = 18, 55

#: The FHIR fields this reads. Nothing else in a bundle is touched.
_RXNORM = "rxnorm"
_ACTIVE = "active"


def _stream(seed: int, name: str) -> np.random.Generator:
    """An independent random stream per component, keyed by name -- as in build.py, so
    adding a column later cannot silently reshuffle who is on what."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


# --------------------------------------------------------------------------- #
# The crosswalks
# --------------------------------------------------------------------------- #

@cache
def risk_table() -> pl.DataFrame:
    """VA drug class -> CDC mechanism, hazard, weight, ACB and the three flags."""
    return pl.read_csv(REFERENCE / "med_climate_risk.csv")


@cache
def _risk() -> dict[str, dict]:
    return {r["va_class_id"]: r for r in risk_table().to_dicts()}


@cache
def _classes_by_rxcui() -> dict[str, tuple[str, ...]]:
    """RxNorm code -> its most specific VA drug class(es).

    The parent/child relation is derived, not declared: class *a* is broader than *b* when
    every member of *b* is also a member of *a*. Whenever a drug resolves to both, only
    *b* survives -- otherwise "hydrochlorothiazide" would score as a thiazide and again as
    a diuretic.
    """
    xw = pl.read_parquet(REFERENCE / "va_drug_class_members.parquet")
    members: dict[str, set[str]] = {}
    per_drug: dict[str, set[str]] = {}
    for rxcui, class_id in zip(xw["rxcui"], xw["va_class_id"], strict=True):
        members.setdefault(class_id, set()).add(rxcui)
        per_drug.setdefault(rxcui, set()).add(class_id)

    broader = {c: {p for p, m in members.items() if p != c and members[c] < m} for c in members}
    return {rxcui: tuple(sorted(cs - {p for c in cs for p in broader[c]}))
            for rxcui, cs in per_drug.items()}


def classes_for(rxcuis: Iterable[str]) -> list[str]:
    """The distinct VA drug classes these active codes resolve to, most specific only.

    A code outside the crosswalk contributes nothing: statins, vitamins and contraceptives
    are real active medications with no climate mechanism, and must not invent a flag.
    """
    lookup = _classes_by_rxcui()
    return sorted({c for rxcui in rxcuis for c in lookup.get(str(rxcui), ())})


# --------------------------------------------------------------------------- #
# Derivation -- docs/SPEC.md 5.3
# --------------------------------------------------------------------------- #

def flags_for(rxcuis: Sequence[str]) -> dict:
    """The whole medication block for one veteran's active code list."""
    risk = _risk()
    cs = classes_for(rxcuis)
    combo = (any(c in RAAS_CLASSES for c in cs) and any(c in DIURETIC_CLASSES for c in cs))
    return {
        "va_drug_classes": cs,
        "n_active_meds": len(rxcuis),
        "med_thermoreg_score": round(
            sum(risk[c]["weight"] for c in cs if risk[c]["hazard"] == "heat"), 4),
        "acb_score": sum(risk[c]["acb"] for c in cs),
        "med_combo_raas_diuretic": combo,
        "med_renal_triple": combo and any(c in NSAID_CLASSES for c in cs),
        "med_cold_chain": any(risk[c]["cold_chain"] for c in cs),
        "med_controlled": any(risk[c]["controlled"] for c in cs),
        "med_narrow_ti": any(risk[c]["narrow_ti"] for c in cs),
    }


def derive(rxcui_lists: Iterable[Sequence[str]]) -> pl.DataFrame:
    """One row of derived medication columns per veteran, in the order given."""
    rows = [flags_for(list(rx or ())) for rx in rxcui_lists]
    return pl.DataFrame(rows, schema={
        "va_drug_classes": pl.List(pl.Utf8),
        "n_active_meds": pl.Int32,
        "med_thermoreg_score": pl.Float64,
        "acb_score": pl.Int32,
        "med_combo_raas_diuretic": pl.Boolean,
        "med_renal_triple": pl.Boolean,
        "med_cold_chain": pl.Boolean,
        "med_controlled": pl.Boolean,
        "med_narrow_ti": pl.Boolean,
    })


# --------------------------------------------------------------------------- #
# The prescription profiles
# --------------------------------------------------------------------------- #

_SEX = {"male": "M", "female": "F"}


def profiles_from_fhir(zip_path: Path) -> pl.DataFrame:
    """Distil Synthea's FHIR sample down to one row per bundle.

    Kept: the patient's *active* RxNorm codes, their sex, and their age at the last dated
    event in their own record -- intrinsic to the bundle, so the table is reproducible.
    Everything else in the 30 MB of FHIR is dropped, which is why the result can be
    committed and a clean clone still gets real medication lists.
    """
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            bundle = json.loads(z.read(name))
            patient_id = birth = sex = last = None
            rxcuis: set[str] = set()
            for entry in bundle.get("entry", ()):
                res = entry.get("resource", {})
                kind = res.get("resourceType")
                if kind == "Patient" and patient_id is None:
                    patient_id = res.get("id")
                    birth = res.get("birthDate")
                    sex = _SEX.get(res.get("gender"))
                elif kind == "MedicationRequest":
                    last = max(last or "", (res.get("authoredOn") or "")[:10]) or None
                    if res.get("status") == _ACTIVE:
                        rxcuis |= _rxnorm_codes(res)
                elif kind == "Encounter":
                    last = max(last or "", (res.get("period", {}).get("start") or "")[:10]) or None
            if patient_id is None or birth is None or last is None:
                continue        # practitioner and hospital rosters carry no Patient
            rows.append({
                "profile_id": patient_id,
                "age": (date.fromisoformat(last) - date.fromisoformat(birth)).days // 365,
                "sex": sex,
                "rxcuis": sorted(rxcuis),
            })
    return pl.DataFrame(rows, schema={"profile_id": pl.Utf8, "age": pl.Int32,
                                      "sex": pl.Utf8, "rxcuis": pl.List(pl.Utf8)})


def _rxnorm_codes(resource: dict) -> set[str]:
    """RxNorm codes on a MedicationRequest, whichever way the version spells it."""
    concept = resource.get("medicationCodeableConcept") or \
        (resource.get("medication") or {}).get("concept") or {}
    return {str(c["code"]) for c in concept.get("coding", ())
            if _RXNORM in (c.get("system") or "").lower() and c.get("code")}


@cache
def load_profiles() -> pl.DataFrame:
    return pl.read_parquet(REFERENCE / "synthea_med_profiles.parquet")


def prescribe(ages: Iterable[int], seed: int = 0) -> list[list[str]]:
    """Give every veteran a real active-medication list, drawn within their age band."""
    profiles = load_profiles().filter(pl.col("age") >= ADULT_AGE)
    pools = {
        False: profiles.filter(pl.col("age") < OLDER_PROFILE_AGE)["rxcuis"].to_list(),
        True: profiles.filter(pl.col("age") >= OLDER_PROFILE_AGE)["rxcuis"].to_list(),
    }
    ages = list(ages)
    draw = _stream(seed, "prescribe").random(len(ages))
    out = []
    for age, u in zip(ages, draw, strict=True):
        pool = pools[age >= OLDER_PROFILE_AGE]
        out.append(list(pool[int(u * len(pool))]))
    return out


def supply(mail_order: np.ndarray, seed: int = 0) -> np.ndarray:
    """Days of supply left today: uniform phase through a 90-day mail or 30-day window fill."""
    fill = np.where(mail_order, MAIL_FILL_DAYS, WINDOW_FILL_DAYS)
    u = _stream(seed, "supply_phase").random(mail_order.size)
    return np.floor(u * (fill + 1)).astype(np.int32)


def attach(people: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Add every medication column in the cohort contract to `people`."""
    rxcuis = prescribe(people["age"].to_list(), seed)
    mail = _stream(seed, "mail_order").random(people.height) < MAIL_ORDER_RATE
    return people.with_columns(
        pl.Series("med_rxcuis", rxcuis, dtype=pl.List(pl.Utf8)),
        *derive(rxcuis).get_columns(),
        pl.Series("mail_order_pharmacy", mail),
        pl.Series("days_supply_remaining", supply(mail, seed), dtype=pl.Int32),
    )


# --------------------------------------------------------------------------- #
# Entry point: what the layer does, in one screen
# --------------------------------------------------------------------------- #

def main() -> int:
    profiles = load_profiles()
    on_meds = profiles.filter(pl.col("rxcuis").list.len() > 0)
    d = derive(on_meds["rxcuis"].to_list())
    print(f"{profiles.height} Synthea bundles, {on_meds.height} with an active medication\n")
    print(f"  {'on any climate-relevant medication':44s} "
          f"{d['va_drug_classes'].list.len().gt(0).mean():6.1%}")
    for label, series in (
            ("impairs heat response (thermoreg score > 0)", d["med_thermoreg_score"] > 0),
            ("the CDC-named pair: RAAS + diuretic", d["med_combo_raas_diuretic"]),
            ("anticholinergic burden >= 3", d["acb_score"] >= 3),
            ("controlled: retail refill excludes these", d["med_controlled"]),
            ("cold chain: spoils in an outage", d["med_cold_chain"]),
            ("narrow therapeutic index", d["med_narrow_ti"]),
            ("NSAID on top of RAAS + diuretic", d["med_renal_triple"])):
        print(f"  {label:44s} {series.mean():6.1%}")
    print("\nLeeward flags these for the VA clinical pharmacist. It never changes a dose.")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
