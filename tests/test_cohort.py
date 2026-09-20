"""Acceptance tests for `leeward/cohort/build.py` -- the 10,000-veteran panel.

Written before the module. The claim they protect is the one said on stage: *every
neighbourhood-level rate in the cohort is real and cited; only the people are synthetic.*
So each test checks the cohort against `data/reference/` directly, never against tables
that build.py computed for itself -- a wrong join in build.py cannot also fix its own test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from leeward import schema
from leeward.cohort import build

REF = schema.REFERENCE
N = 10_000

# cohort field -> the CDC PLACES column its per-ZIP rate comes from (docs/SPEC.md §5.3).
PLACES_FIELDS = {
    "mobility_impaired": "mobility_crudeprev",
    "low_assets": "shututility_crudeprev",
    "transport_barrier": "lacktrpt_crudeprev",
    "caregiver_none": "emotionspt_crudeprev",
    "copd": "copd_crudeprev",
    "asthma": "casthma_crudeprev",
    "active_cancer_tx": "cancer_crudeprev",
    "depression": "depression_crudeprev",
    "diabetes": "diabetes_crudeprev",
    # SNAP receipt: a measured per-ZIP floor on the low-income share.
    "income_low": "foodstamp_crudeprev",
}


@pytest.fixture(scope="module")
def cohort() -> pl.DataFrame:
    return build.build(n=N, seed=0).with_columns(
        (pl.col("caregiver") == "none").alias("caregiver_none"),
        (pl.col("income_band") == "low").alias("income_low"))


def _members() -> pl.DataFrame:
    """MODZCTA -> member ZCTA, one row each. ACS and emPOWER are published per ZCTA/ZIP."""
    mz = pl.read_parquet(REF / "nyc_modzcta.parquet")
    return (mz.select("modzcta", pl.col("zcta_members").str.split(",").alias("zcta"))
              .explode("zcta", empty_as_null=True).with_columns(pl.col("zcta").str.strip_chars()))


def _acs_by_modzcta() -> pl.DataFrame:
    acs = pl.read_parquet(REF / "acs_veterans_by_zcta.parquet")
    num = [c for c in acs.columns if c != "zcta"]
    return (acs.join(_members(), on="zcta", how="inner")
               .group_by("modzcta").agg(pl.col(num).sum()))


# --------------------------------------------------------------------------- #
# Shape and geography
# --------------------------------------------------------------------------- #

def test_ten_thousand_rows_that_satisfy_the_contract(cohort: pl.DataFrame) -> None:
    assert cohort.height == N
    assert cohort["veteran_id"].n_unique() == N
    schema.validate(cohort, "cohort")


def test_every_modzcta_is_one_of_the_real_178(cohort: pl.DataFrame) -> None:
    real = set(pl.read_parquet(REF / "nyc_modzcta.parquet")["modzcta"].to_list())
    assert len(real) == 178
    got = set(cohort["modzcta"].to_list())
    assert got <= real, f"ZIPs that are not NYC MODZCTAs: {sorted(got - real)[:5]}"
    assert len(got) > 150, f"10,000 veterans reached only {len(got)} ZIPs"


def test_zip_frequencies_track_acs_veteran_counts(cohort: pl.DataFrame) -> None:
    """Re-homing is P(zip | age band) ∝ ACS veterans, so counts must follow ACS."""
    acs = _acs_by_modzcta().select("modzcta", "veterans_total")
    freq = cohort.group_by("modzcta").len()
    j = acs.join(freq, on="modzcta", how="left").fill_null(0)
    r = np.corrcoef(j["veterans_total"].to_numpy(), j["len"].to_numpy())[0, 1]
    assert r > 0.7, f"ZIP frequencies correlate {r:.2f} with ACS veteran counts"


def test_age_and_sex_follow_acs(cohort: pl.DataFrame) -> None:
    """B21001 is sex by age by veteran status, so both margins are real."""
    acs = pl.read_parquet(REF / "acs_veterans_by_zcta.parquet")
    total = acs["veterans_total"].sum()
    share_65 = acs["veterans_65plus"].sum() / total
    share_f = sum(acs[c].sum() for c in acs.columns if c.startswith("vet_f_")) / total

    got_65 = (cohort["age"] >= 65).mean()
    got_f = (cohort["sex"] == "F").mean()
    assert abs(got_65 - share_65) < 0.02, f"65+ share {got_65:.3f}, ACS says {share_65:.3f}"
    assert abs(got_f - share_f) < 0.015, f"female share {got_f:.3f}, ACS says {share_f:.3f}"


# --------------------------------------------------------------------------- #
# Neighbourhood rates are read, not invented
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,col", sorted(PLACES_FIELDS.items()))
def test_places_rate_within_20pct_of_zip_weighted_source(
        cohort: pl.DataFrame, field: str, col: str) -> None:
    places = pl.read_parquet(REF / "places_zcta_nyc.parquet").select(
        pl.col("zcta").alias("modzcta"), col)
    j = cohort.join(places, on="modzcta", how="left")
    assert j[col].null_count() == 0, f"veterans in ZIPs with no PLACES {col}"
    realised = j[field].mean()
    source = j[col].mean() / 100  # each veteran carries their own ZIP's rate
    assert abs(realised / source - 1) < 0.20, (
        f"{field}: realised {realised:.2%}, ZIP-weighted PLACES {col} {source:.2%}")


def test_places_gradient_survives_into_the_cohort(cohort: pl.DataFrame) -> None:
    """A citywide constant would pass the 20% test above. It would not pass this one.

    PLACES puts utility-shutoff threat at 5.3% in HVI-1 ZIPs and 17.2% in HVI-5 ZIPs.
    If the cohort flattens that, the join is wrong even if the average is right.
    """
    by = cohort.group_by("hvi").agg(
        pl.col("low_assets", "mobility_impaired", "transport_barrier").mean())
    lo = by.filter(pl.col("hvi") == 1).row(0, named=True)
    hi = by.filter(pl.col("hvi") == 5).row(0, named=True)
    for f in ("low_assets", "mobility_impaired", "transport_barrier"):
        assert hi[f] > 1.5 * lo[f], f"{f}: HVI 5 {hi[f]:.1%} vs HVI 1 {lo[f]:.1%}"


def test_powered_equipment_follows_empower(cohort: pl.DataFrame) -> None:
    """emPOWER counts electricity-dependent Medicare beneficiaries; ÷ ACS 65+ is the rate."""
    emp = (pl.read_parquet(REF / "empower_ny_zip.parquet")
             .join(_members(), left_on="zip", right_on="zcta", how="inner")
             .group_by("modzcta").agg(pl.col("dme_power_dependent").sum()))
    rate = (_acs_by_modzcta().join(emp, on="modzcta")
            .select("modzcta", (pl.col("dme_power_dependent") / pl.col("pop_65plus"))
                    .alias("rate")))
    older = cohort.filter(pl.col("age") >= 65).join(rate, on="modzcta", how="left")
    powered = older["powered_equipment"].is_in(["oxygen", "ventilator", "wheelchair", "bed"])
    realised, source = powered.mean(), older["rate"].mean()
    assert abs(realised / source - 1) < 0.25, (
        f"65+ powered equipment {realised:.2%}, ZIP-weighted emPOWER {source:.2%}")


# --------------------------------------------------------------------------- #
# Dialysis: the level is the VA's own, the geography is emPOWER's
# --------------------------------------------------------------------------- #

# ESRD prevalence among VA-enrolled veterans: 604 per 100,000. Wang et al., BMC Health
# Serv Res 2013;13:26, https://doi.org/10.1186/1472-6963-13-26 (docs/sources.md). Written out
# here rather than imported, so the test cannot agree with a wrong constant in build.py.
VA_ESRD_PER_100K = 604


def test_dialysis_prevalence_is_the_vas_own_esrd_rate(cohort: pl.DataFrame) -> None:
    """emPOWER counts only Medicare *facility* dialysis, which put 7 of 10,000 veterans on it
    (0.07%). The VA's own figure is 0.604%: about 60 of 10,000. Under 40 or over 85 is more
    than 3 standard deviations from it."""
    n = cohort["ckd_dialysis"].sum()
    assert 40 <= n <= 85, (
        f"{n} of {N:,} veterans on dialysis; the VA's ESRD rate of {VA_ESRD_PER_100K} per "
        f"100,000 puts it near {N * VA_ESRD_PER_100K / 100_000:.0f}")


def test_dialysis_probability_takes_its_level_from_the_va_and_its_shape_from_empower() -> None:
    """Flattening to a citywide 0.604% would pass the count test above and throw away a
    measured per-ZIP pattern. So: the mean is exactly the VA's, and within an age group each
    veteran's probability stays proportional to their own ZIP's emPOWER rate."""
    people = build.rehome(N, 0, build.zip_frame())
    rate, age = people["rate_dialysis"].to_numpy(), people["age"].to_numpy()
    p = build.dialysis_probability(rate, age)

    assert abs(p.mean() - VA_ESRD_PER_100K / 100_000) < 1e-9
    assert p.min() >= 0 and p.max() < 0.10, "rescaling made some ZIP's rate implausible"
    for group in (age >= 65, age < 65):
        keep = group & (rate > 0)
        assert np.allclose(p[keep] / rate[keep], (p[keep] / rate[keep])[0]), (
            "probability is no longer proportional to the ZIP's emPOWER rate")
    older, younger = (age >= 65) & (rate > 0), (age < 65) & (rate > 0)
    assert (p[younger] / rate[younger]).mean() < (p[older] / rate[older]).mean(), (
        "under-65s should sit below 65+ at the same emPOWER rate; emPOWER counts Medicare")


def test_the_sandy_dialysis_story_is_more_than_a_handful_of_people(cohort: pl.DataFrame) -> None:
    """It rested on seven people. Station 630 is the campus that evacuated in 2012, and at
    least five of the veterans sent there for dialysis are what makes that scenario land."""
    on = cohort.filter(pl.col("ckd_dialysis"))
    assert on.height >= 40
    at_630 = on.filter(pl.col("facility_id") == "630").height
    assert at_630 >= 5, f"only {at_630} dialysis veterans at station 630"


def test_the_va_esrd_rate_is_cited_with_its_url() -> None:
    """CLAUDE.md: every real number shown anywhere is in docs/sources.md with a URL."""
    text = (Path(__file__).resolve().parents[1] / "docs" / "sources.md").read_text("utf-8")
    assert build.VA_ESRD_PER_100K == VA_ESRD_PER_100K
    assert f"{VA_ESRD_PER_100K} vs. 187 per 100,000" in text, (
        "the paper's sentence giving the VA ESRD rate is not quoted in docs/sources.md")
    assert "10.1186/1472-6963-13-26" in text, "the ESRD rate has no URL in docs/sources.md"


# --------------------------------------------------------------------------- #
# Race and ethnicity: drawn from the ZIP's measured composition, never invented
# --------------------------------------------------------------------------- #

RACES = {"White", "Black", "Asian", "Other"}
ETHNICITIES = {"Hispanic", "Non-Hispanic"}

# B03002 cell -> the cohort value it must feed. Written out here, not imported from
# build.py, so a wrong mapping in the build cannot also fix its own test.
SOURCE_RACE = {"White": ["nh_white", "hisp_white"], "Black": ["nh_black", "hisp_black"],
               "Asian": ["nh_asian", "hisp_asian"],
               "Other": ["nh_aian", "nh_nhpi", "nh_other", "nh_multi",
                         "hisp_aian", "hisp_nhpi", "hisp_other", "hisp_multi"]}
SOURCE_HISPANIC = ["hisp_white", "hisp_black", "hisp_aian", "hisp_asian", "hisp_nhpi",
                   "hisp_other", "hisp_multi"]


def _race_shares() -> pl.DataFrame:
    """Per-MODZCTA share of residents in each cohort race, and of Hispanic origin, straight
    from the vendored B03002 table."""
    acs = pl.read_parquet(REF / "acs_race_by_zcta.parquet")
    num = [c for c in acs.columns if c != "zcta"]
    by_zip = (acs.join(_members(), on="zcta", how="inner")
                 .group_by("modzcta").agg(pl.col(num).sum()))
    return by_zip.select(
        "modzcta",
        *[(pl.sum_horizontal(cells) / pl.col("pop_total")).alias(f"src_{race}")
          for race, cells in SOURCE_RACE.items()],
        (pl.sum_horizontal(SOURCE_HISPANIC) / pl.col("pop_total")).alias("src_Hispanic"),
        (pl.col("nh_white") / pl.col("pop_total")).alias("src_nh_white"))


def test_race_and_ethnicity_are_filled_and_say_they_are_synthetic(cohort: pl.DataFrame) -> None:
    """They were null for all 10,000, which left the fairness audit nothing to stratify by."""
    assert cohort["race"].null_count() == 0
    assert cohort["ethnicity"].null_count() == 0
    assert set(cohort["race"].unique().to_list()) == RACES
    assert set(cohort["ethnicity"].unique().to_list()) == ETHNICITIES
    assert cohort["race_synthetic"].all() and cohort["ethnicity_synthetic"].all(), (
        "a real ZIP composition does not make an individual's race real")


@pytest.mark.parametrize("group", ["White", "Black", "Asian", "Other", "Hispanic"])
def test_race_share_matches_the_composition_of_the_zips_veterans_live_in(
        cohort: pl.DataFrame, group: str) -> None:
    """Each veteran carries their own ZIP's composition, so the cohort's share of a group is
    the mean of its veterans' ZIP shares -- as the PLACES rates are tested above."""
    j = cohort.join(_race_shares(), on="modzcta", how="left")
    assert j[f"src_{group}"].null_count() == 0, "veterans in ZIPs with no B03002 row"
    col = "ethnicity" if group == "Hispanic" else "race"
    realised = (j[col] == group).mean()
    source = j[f"src_{group}"].mean()
    assert abs(realised / source - 1) < 0.10, (
        f"{group}: realised {realised:.1%}, ZIP-weighted ACS B03002 {source:.1%}")


def test_race_gradient_survives_into_the_cohort(cohort: pl.DataFrame) -> None:
    """A citywide [55/27/8/10] split would pass the average test above. It cannot pass this:
    where ACS says a ZIP is mostly Black, the cohort's veterans there must be mostly Black."""
    j = cohort.join(_race_shares(), on="modzcta", how="left")
    hi = j.filter(pl.col("src_Black") >= 0.5)
    lo = j.filter(pl.col("src_Black") <= 0.1)
    assert hi.height >= 200 and lo.height >= 200, "too few veterans to compare the two ends"
    assert (hi["race"] == "Black").mean() > 0.4, "majority-Black ZIPs came out under 40% Black"
    assert (lo["race"] == "Black").mean() < 0.15, "ZIPs under 10% Black came out over 15% Black"


def test_minority_share_agrees_with_cdc_svi_by_borough(cohort: pl.DataFrame) -> None:
    """An independent check on a different source. CDC SVI `EP_MINRTY` is the share of
    residents who are anything but non-Hispanic white, per census tract. Population-weighted
    to borough it must land near the cohort's share, or B03002 was read wrongly."""
    svi = (pl.read_parquet(REF / "svi_nyc_tract.parquet")
             .filter((pl.col("EP_MINRTY") >= 0) & (pl.col("E_TOTPOP") > 0))
             .group_by("borough")
             .agg(((pl.col("EP_MINRTY") / 100) * pl.col("E_TOTPOP")).sum().alias("m"),
                  pl.col("E_TOTPOP").sum().alias("pop")))
    svi = svi.with_columns((pl.col("m") / pl.col("pop")).alias("svi_minority"))
    ours = cohort.group_by("borough").agg(
        (~((pl.col("race") == "White") & (pl.col("ethnicity") == "Non-Hispanic")))
        .mean().alias("minority"))
    j = svi.join(ours, on="borough", how="inner")
    assert j.height == 5
    for row in j.iter_rows(named=True):
        assert abs(row["minority"] - row["svi_minority"]) < 0.10, (
            f"{row['borough']}: cohort {row['minority']:.1%} non-white-non-Hispanic, "
            f"CDC SVI says {row['svi_minority']:.1%}")


def test_hazard_exposure_is_the_real_zip_join(cohort: pl.DataFrame) -> None:
    evac = pl.read_parquet(REF / "evac_zone_by_modzcta.parquet").select("modzcta", "evac_zone_min")
    storm = pl.read_parquet(REF / "stormwater_by_modzcta.parquet").rename(
        {"stormwater_flooded_frac": "storm_ref"})
    hvi = pl.read_parquet(REF / "hvi_by_zcta.parquet").select(
        pl.col("zcta").alias("modzcta"), pl.col("hvi").alias("hvi_ref"))
    j = cohort.join(evac, on="modzcta").join(storm, on="modzcta").join(hvi, on="modzcta")
    assert j.height == N
    assert (j["evac_zone"] == j["evac_zone_min"]).all()
    assert ((j["stormwater_flooded_frac"] - j["storm_ref"]).abs() < 1e-9).all()
    assert (j["hvi"] == j["hvi_ref"]).all()


# --------------------------------------------------------------------------- #
# Facility assignment -- the SiteDown term depends on it
# --------------------------------------------------------------------------- #

def test_facilities_are_care_sites_and_site_dependent_care_goes_where_it_is_given(
        cohort: pl.DataFrame) -> None:
    fac = pl.read_parquet(REF / "va_facilities_nyc_hazard.parquet")
    # Vet Centers counsel rather than treat, and the mobile clinic has no fixed site.
    care = fac.filter(~pl.col("station_no").str.ends_with("V")
                      & ~pl.col("name").str.contains("Mobile"))
    assert set(cohort["facility_id"].to_list()) <= set(care["station_no"].to_list())

    dialysis_sites = set(fac.filter(pl.col("site_dependent_services"))["station_no"].to_list())
    on_dialysis = cohort.filter(pl.col("ckd_dialysis"))
    assert set(on_dialysis["facility_id"].to_list()) <= dialysis_sites

    otp = cohort.filter(pl.col("on_methadone_otp"))
    assert otp.height > 0
    assert set(otp["facility_id"].to_list()) <= {"630", "630A4"}, "OTP is Manhattan or Brooklyn"
    assert otp.filter(pl.col("facility_id") == "630").height > 0, (
        "no OTP patients at station 630, so the Sandy SiteDown story has nobody in it")


def test_everyone_else_goes_to_the_nearest_care_site(cohort: pl.DataFrame) -> None:
    """Staten Island has one VA clinic. Everyone there without site-dependent care uses it."""
    si = cohort.filter((pl.col("borough") == "Staten Island")
                       & ~pl.col("ckd_dialysis") & ~pl.col("on_methadone_otp"))
    assert si.height > 0
    assert (si["facility_id"] == "630GB").all()


# --------------------------------------------------------------------------- #
# Reproducibility and the write path
# --------------------------------------------------------------------------- #

def test_same_seed_same_cohort_different_seed_different_cohort() -> None:
    a, b = build.build(n=500, seed=0), build.build(n=500, seed=0)
    assert a.equals(b), "the same seed must produce the same cohort, byte for byte"
    assert not a.equals(build.build(n=500, seed=1))


def test_cli_defaults_are_ten_thousand_veterans_and_seed_zero() -> None:
    args = build.parse_args([])
    assert args.n == N and args.seed == 0


def test_writes_through_the_schema_validator() -> None:
    src = inspect.getsource(build)
    assert 'schema.write(' in src, "write with schema.write(df, 'cohort'), which validates first"
    assert "write_parquet" not in src, "never write a contract table with write_parquet directly"
