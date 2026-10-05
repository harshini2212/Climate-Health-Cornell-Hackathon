"""The two real heat-outcome fetchers, parsed from small in-memory payloads. No network.

The fetch itself is one GET per file; everything that can be wrong about the result -- two
date formats, two column orders, SPARCS suppression, the UHF42 and CD -> modzcta mapping -- lives in
the pure `parse` functions, so that is what these pin.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from leeward.ingest.sources import ehdp_heat, heat_syndrome

REF = Path(__file__).resolve().parents[2] / "data" / "reference"

SUPP = "END_DATE,MAX_DAILY_TEMP,HEAT_ED_VISIT_COUNT\n7/1/2020,91,12\n7/2/2020,84,3\n"
LIVE = '"END_DATE","HEAT_ED_VISIT_COUNT","MAX_DAILY_TEMP"\n2021-07-01,9,90\n'


def test_heat_syndrome_reads_both_files_whatever_their_column_order() -> None:
    df = heat_syndrome.parse({"edheat1720_supp.csv": SUPP, "edheat2021_live.csv": LIVE})
    assert df.columns == ["date", "max_temp_f", "heat_ed_visits", "file", "source"]
    assert df["date"].to_list() == [date(2020, 7, 1), date(2020, 7, 2), date(2021, 7, 1)]
    assert df["max_temp_f"].to_list() == [91.0, 84.0, 90.0]
    assert df["heat_ed_visits"].to_list() == [12, 3, 9]
    assert set(df["source"]) == {heat_syndrome.SOURCE}


def test_heat_syndrome_refuses_a_day_published_twice() -> None:
    with pytest.raises(ValueError, match="more than once"):
        heat_syndrome.parse({"a.csv": SUPP, "b.csv": SUPP})


def test_heat_syndrome_refuses_a_negative_count() -> None:
    bad = "END_DATE,MAX_DAILY_TEMP,HEAT_ED_VISIT_COUNT\n7/1/2020,91,-1\n"
    with pytest.raises(ValueError, match="negative"):
        heat_syndrome.parse({"a.csv": bad})


META = [{"IndicatorID": 2075, "IndicatorName": "Heat stress: yearly emergency department visits",
         "Measures": [{"MeasureID": 535, "MeasurementType": "Number"},
                      {"MeasureID": 536, "MeasurementType": "Estimated annual rate"}]}]
PERIODS = [{"TimePeriodID": 45, "TimePeriod": "2018"}, {"TimePeriodID": 312,
                                                         "TimePeriod": "2017-2021"}]
GEO = pl.DataFrame({"GeoType": ["UHF42", "UHF42"], "GeoID": [101, 102],
                    "Name": ["Kingsbridge - Riverdale", "Northeast Bronx"]})
SUPPRESSED = "** Numbers <11 are suppressed as required by SPARCS"
PAYLOAD = {"MeasureID": [535, 536, 536], "GeoID": [101, 101, 102],
           "GeoType": ["UHF42", "UHF42", "UHF42"], "TimePeriodID": [45, 45, 312],
           "Value": [None, 7.5, 12.0], "CI": ["", "", ""], "Note": [SUPPRESSED, "", ""],
           "DisplayValue": ["**", "7.5", "12.0"]}


def test_ehdp_parse_keeps_suppression_as_null_not_zero() -> None:
    df = ehdp_heat.parse({2075: PAYLOAD}, META, PERIODS, GEO)
    assert df.height == 3
    first = df.row(0, named=True)
    assert first["value"] is None and first["suppressed"]
    assert first["outcome"] == "heat_ed_visits" and first["measure"] == "Number"
    assert first["geo_name"] == "Kingsbridge - Riverdale" and first["geo_id"] == "101"
    assert df["year"].to_list() == [2018, 2018, None], "a multi-year period has no single year"
    assert df["time_period"].to_list() == ["2018", "2018", "2017-2021"]
    assert not df["suppressed"][1] and df["value"][1] == 7.5


def test_ehdp_parse_refuses_an_indicator_the_metadata_does_not_describe() -> None:
    with pytest.raises(ValueError, match="2076"):
        ehdp_heat.parse({2076: PAYLOAD}, META, PERIODS, GEO)


def test_crosswalk_takes_the_majority_uhf_and_the_largest_cd_overlap() -> None:
    zcta_to_uhf = pl.DataFrame({"uhfcode": [101, 101, 102, 103],
                                "zcta": ["10463", "10471", "10466", "10001"]})
    modzcta = pl.DataFrame({"modzcta": ["10463", "10466", "10001"],
                            "zcta_members": ["10463, 10471", "10466", "10001, 10119"]})
    overlap = pl.DataFrame({"modzcta": ["10463", "10463", "10466", "10001"],
                            "cd": [208, 207, 212, 104], "area": [9.0, 1.0, 5.0, 3.0]})
    xw = ehdp_heat.crosswalk(zcta_to_uhf, modzcta, overlap)
    assert xw.sort("modzcta").rows() == [("10001", "103", "104"), ("10463", "101", "208"),
                                         ("10466", "102", "212")]


def test_crosswalk_refuses_a_modzcta_it_cannot_place() -> None:
    zcta_to_uhf = pl.DataFrame({"uhfcode": [101], "zcta": ["10463"]})
    modzcta = pl.DataFrame({"modzcta": ["10463", "99999"], "zcta_members": ["10463", "99999"]})
    overlap = pl.DataFrame({"modzcta": ["10463"], "cd": [208], "area": [1.0]})
    with pytest.raises(ValueError, match="99999"):
        ehdp_heat.crosswalk(zcta_to_uhf, modzcta, overlap)


def test_the_committed_crosswalk_places_every_modzcta_in_a_real_cd() -> None:
    """data/reference is committed input; the overlay that built this needs geopandas."""
    xw = pl.read_parquet(REF / f"{ehdp_heat.XWALK_STEM}.parquet")
    modzcta = pl.read_parquet(REF / "nyc_modzcta.parquet")
    assert sorted(xw["modzcta"]) == sorted(modzcta["modzcta"])
    cds = pl.read_parquet(REF / f"{ehdp_heat.STEM}.parquet").filter(
        pl.col("geo_type") == "CD")["geo_id"].unique()
    assert set(xw["cd"]) <= set(cds), "a modzcta landed in a CD EHDP does not report"
    assert xw["cd"].n_unique() >= 55, "most of the 59 districts should hold some ZIP"
