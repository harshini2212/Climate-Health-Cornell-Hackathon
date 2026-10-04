"""The contract file must be internally consistent, and `empty()` must satisfy `validate()`.

If this breaks, every lane is building against a contract that cannot be satisfied.
"""

from __future__ import annotations

import polars as pl
import pytest

from leeward import schema
from leeward.schema import ACTION_COST_UNIT, DEFAULT_CAPACITY, NEEDS, TIERS, SchemaError


@pytest.mark.parametrize("name", sorted(schema.TABLES))
def test_empty_frame_satisfies_its_own_contract(name: str) -> None:
    schema.validate(schema.empty(name), name)


@pytest.mark.parametrize("name", sorted(schema.TABLES))
def test_key_columns_exist_and_are_not_nullable(name: str) -> None:
    t = schema.TABLES[name]
    for k in t.key:
        col = t.column(k)
        assert col is not None, f"{name}: key column {k} is not declared"
        assert not col.nullable, f"{name}: key column {k} must not be nullable"


@pytest.mark.parametrize("name", sorted(schema.TABLES))
def test_no_duplicate_column_names(name: str) -> None:
    names = [c.name for c in schema.TABLES[name].columns]
    assert len(names) == len(set(names)), f"{name} declares a column twice"


def test_every_action_has_a_capacity_bucket_with_a_budget() -> None:
    for action, bucket in ACTION_COST_UNIT.items():
        assert bucket in DEFAULT_CAPACITY, f"{action} costs {bucket!r}, which has no budget"


def test_vocabularies_are_the_ones_the_docs_promise() -> None:
    assert NEEDS == ["breathing", "heat", "mental", "treatment_gap", "access_loss"]
    assert TIERS == ["act_now", "find_out", "self_serve", "everyday"]


def test_validate_actually_rejects_bad_data() -> None:
    """A validator that never fails is worse than none, because it is trusted."""
    good = schema.empty("scores")

    with pytest.raises(SchemaError, match="missing columns"):
        schema.validate(good.drop("p_mean"), "scores")

    bad_need = pl.DataFrame({
        **{c.name: pl.Series(c.name, [None], dtype=c.dtype) for c in schema.TABLES["scores"].columns},
    }).with_columns([
        pl.lit("v1").alias("veteran_id"), pl.lit("not_a_need").alias("need"),
        pl.lit(0.5).alias("p_mean"), pl.lit(0.4).alias("p_lo80"), pl.lit(0.6).alias("p_hi80"),
        pl.lit(0.2).alias("p_epistemic_share"), pl.lit(0).cast(pl.Int32).alias("model_rung"),
        pl.lit("2026-07-01").str.to_date().alias("date"),
    ])
    with pytest.raises(SchemaError, match="not in"):
        schema.validate(bad_need, "scores")

    out_of_range = bad_need.with_columns(pl.lit("heat").alias("need"), pl.lit(1.7).alias("p_mean"))
    with pytest.raises(SchemaError, match="outside contract"):
        schema.validate(out_of_range, "scores")


def test_synthetic_columns_require_their_flag() -> None:
    df = schema.empty("cohort").drop("home_ac_synthetic")
    with pytest.raises(SchemaError, match="home_ac_synthetic"):
        schema.validate(df, "cohort")


def test_describe_renders_for_every_table() -> None:
    for name in schema.TABLES:
        text = schema.describe(name)
        assert name in text and "key" in text
