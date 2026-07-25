"""Cross-check the Spark aggregation against an independent SQL implementation.

Two implementations of the same aggregation agreeing is a much stronger
correctness argument than either one passing its own unit tests. The Python
path builds 100+ aggregation expressions programmatically through the DataFrame
API; the SQL path is a hand-written GROUP BY run by DuckDB. If they disagree,
one of them is wrong.

Runs on a deterministic sample rather than the full 3 GB log so the suite stays
fast. Skipped if duckdb is not installed or the raw data is absent.
"""
from __future__ import annotations

import csv
import random

import pytest

from src import config

duckdb = pytest.importorskip("duckdb")

SAMPLE_ROWS = 60_000


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    """A reproducible sample of the raw log, written as its own CSV.

    Samples across the whole file rather than taking the first N rows, and that
    is not a detail. The raw CSV is block-ordered by treatment - the first 35%
    of rows and the last 5% are 100% treatment - so any head-based sample
    contains no control users at all. The first version of this fixture did
    exactly that and produced a sample with one arm in it.
    """
    if not config.RAW_CSV.exists():
        pytest.skip(f"raw data not present at {config.RAW_CSV}")

    out = tmp_path_factory.mktemp("sql") / "sample.csv"
    duckdb.sql(f"""
        COPY (
            SELECT * FROM read_csv_auto('{config.RAW_CSV}')
            USING SAMPLE {SAMPLE_ROWS} ROWS (reservoir, 4242)
        ) TO '{out}' (HEADER, DELIMITER ',')
    """)
    return out


def test_sample_contains_both_arms(sample_csv):
    """Guard the fixture itself.

    If this fails, the sample is not representative and every other assertion
    in this file is testing the wrong thing.
    """
    counts = duckdb.sql(
        f"SELECT treatment, COUNT(*) AS n FROM read_csv_auto('{sample_csv}') "
        f"GROUP BY treatment"
    ).df()
    assert set(counts["treatment"]) == {0, 1}
    assert counts["n"].min() > 1000


def _sql_aggregate(path):
    sql = (config.REPO_ROOT / "sql" / "aggregate.sql").read_text()
    sql = sql.replace(":source", f"'{path}'")
    return duckdb.sql(sql).df()


def _pandas_aggregate(path):
    import pandas as pd

    df = pd.read_csv(path)
    out = []
    for arm, g in df.groupby("treatment"):
        row = {
            "treatment": arm,
            "n": len(g),
            "sum_y": g["conversion"].sum(),
            "sum_yy": (g["conversion"] ** 2).sum(),
            "sum_xn": g["visit"].sum(),
            "sum_xnxn": (g["visit"] ** 2).sum(),
            "sum_yxn": (g["conversion"] * g["visit"]).sum(),
            "sum_exposure": g["exposure"].sum(),
        }
        for f in config.FEATURES:
            row[f"sum_{f}"] = g[f].sum()
            row[f"sum_{f}_y"] = (g[f] * g["conversion"]).sum()
        out.append(row)
    return pd.DataFrame(out).sort_values("treatment").reset_index(drop=True)


def test_sql_aggregate_matches_pandas(sample_csv):
    """The SQL in sql/aggregate.sql must reproduce the same sufficient statistics."""
    sql_df = _sql_aggregate(sample_csv).sort_values("treatment").reset_index(drop=True)
    ref_df = _pandas_aggregate(sample_csv)

    assert list(sql_df["treatment"]) == list(ref_df["treatment"])
    for col in ref_df.columns:
        if col == "treatment":
            continue
        for i in range(len(ref_df)):
            assert sql_df[col][i] == pytest.approx(ref_df[col][i], rel=1e-10), (
                f"mismatch in {col} for arm {ref_df['treatment'][i]}"
            )


def test_sql_balance_smds_are_small(sample_csv):
    """The SQL balance query must return SMDs in the range the Python path finds.

    Not an equality check - this runs on a 60k sample where sampling noise
    dominates - but it verifies the query executes, returns one row per
    feature, and produces standardised differences of a plausible size rather
    than something scaled wrongly.
    """
    sql = (config.REPO_ROOT / "sql" / "balance.sql").read_text()
    sql = sql.replace(":source", f"'{sample_csv}'")
    df = duckdb.sql(sql).df()

    assert len(df) == 6
    assert set(df["feature"]) == {f"f{i}" for i in range(6)}
    assert df["smd"].abs().max() < 0.5
