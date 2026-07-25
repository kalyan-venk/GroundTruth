"""Stage 1 - ingest the raw Criteo log in Spark and collapse it to sufficient statistics.

The design idea worth understanding here: every test this pipeline runs -- the
chi-square SRM check, the two-proportion z-test, and CUPED -- depends on the raw
13.9M rows only through a handful of sums. So Spark's job is to compute those
sums, and the driver never sees more than a few dozen numbers.

For the z-test you need n and sum(Y) per arm. For CUPED you need theta =
Cov(Y,X)/Var(X), and both of those come out of n, sum(X), sum(X^2), sum(Y),
sum(XY). All additive. All computable in one pass.

That is what makes "aggregated 13.9M rows" an honest statement rather than a
line that really means "loaded a CSV".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# --- Java -------------------------------------------------------------------

def _resolve_java_home() -> str:
    """Point Spark at JDK 17.

    PySpark 3.5 does not run on JDK 21+: the JVM's strong module encapsulation
    blocks the reflective access Spark's unsafe memory layer relies on, and you
    get an InaccessibleObjectException on sun.nio.ch.DirectBuffer before any of
    your code executes. This machine's default java is 25, so leaving JAVA_HOME
    alone means the pipeline dies at SparkSession.builder.
    """
    if os.environ.get("JAVA_HOME") and "17" in os.environ["JAVA_HOME"]:
        return os.environ["JAVA_HOME"]
    try:
        home = subprocess.run(
            ["/usr/libexec/java_home", "-v", "17"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "JDK 17 not found. PySpark 3.5 will not run on JDK 21+. "
            "Install temurin17 (brew install --cask temurin@17) or set JAVA_HOME."
        )
    os.environ["JAVA_HOME"] = home
    return home


def build_spark():
    _resolve_java_home()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


# --- Schema -----------------------------------------------------------------

def _schema():
    from pyspark.sql.types import (
        StructType, StructField, DoubleType, IntegerType,
    )
    fields = [StructField(f, DoubleType(), True) for f in config.FEATURES]
    fields += [
        StructField("treatment", IntegerType(), True),
        StructField("conversion", IntegerType(), True),
        StructField("visit", IntegerType(), True),
        StructField("exposure", IntegerType(), True),
    ]
    return StructType(fields)


def _read_raw(spark):
    """Read the raw log with an explicit schema.

    inferSchema on a 3 GB CSV costs an extra full pass over the file for no
    benefit -- we already know the types. Declaring them also means a
    malformed row fails loudly instead of silently widening a column to string.
    """
    src = config.RAW_CSV if config.RAW_CSV.exists() else config.RAW_GZ
    if not src.exists():
        raise FileNotFoundError(
            f"No raw data at {config.RAW_CSV}. Run: bash scripts/get_data.sh"
        )
    if src == config.RAW_GZ:
        print("  ! reading the gzip directly - Spark cannot split it, so this "
              "runs single-threaded. Run scripts/get_data.sh to decompress.")
    return (
        spark.read
        .option("header", True)
        .option("mode", "FAILFAST")
        .schema(_schema())
        .csv(str(src))
    )


# --- The pre-assignment covariate ------------------------------------------

def _fit_covariate_model(df, seed: int = 7):
    """Build a pre-randomisation covariate for CUPED out of f0..f11.

    CUPED needs a covariate that is (a) correlated with the outcome and (b)
    unaffected by treatment. The kickstart suggests `visit`, but `visit` is
    measured during the experiment and is itself moved by the ad -- adjusting
    on it would bias the very effect we are trying to estimate. The f-features
    are user attributes fixed before assignment, so they are safe.

    Twelve weak covariates are awkward to use directly, so we collapse them
    into one: fit conversion ~ f0..f11 by OLS and use the fitted value as X.
    This is the standard "CUPED with a learned covariate" trick (Guo et al.'s
    MLRATE). A linear probability model on a binary outcome is not calibrated,
    but calibration is irrelevant here -- we only need something correlated
    with Y, and theta rescales it anyway.

    Fitted on a *held-out sample of control rows only*, for two reasons:
    control-only keeps the treatment effect out of the coefficients, and
    holding out removes any worry that the covariate is overfit to the rows it
    later adjusts.
    """
    fit_sample = (
        df.filter("treatment = 0")
          .select(*config.FEATURES, "conversion")
          .sample(withReplacement=False, fraction=0.25, seed=seed)
          .toPandas()
    )
    X = fit_sample[config.FEATURES].to_numpy(dtype=np.float64)
    y = fit_sample["conversion"].to_numpy(dtype=np.float64)
    X1 = np.column_stack([np.ones(len(X)), X])

    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "intercept": float(beta[0]),
        "coefficients": {f: float(b) for f, b in zip(config.FEATURES, beta[1:])},
        "fit_rows": int(len(fit_sample)),
        "fit_r2": r2,
        "fit_seed": seed,
        "fit_population": "control arm only, 25% sample",
    }


def _covariate_expr(model):
    from pyspark.sql import functions as F

    expr = F.lit(model["intercept"])
    for feat, coef in model["coefficients"].items():
        expr = expr + F.lit(coef) * F.col(feat)
    return expr


# --- The aggregation --------------------------------------------------------

def _aggregate(df):
    """One pass, per arm, producing every sum the stats layer needs.

    Y (conversion) and the naive covariate (visit) are binary, so sum == sum of
    squares for them, but we compute both anyway rather than hard-coding an
    assumption that a future metric change would quietly break.
    """
    from pyspark.sql import functions as F

    Y = F.col("conversion").cast("double")
    Xn = F.col("visit").cast("double")
    Xs = F.col("f_index")

    return (
        df.groupBy("treatment")
        .agg(
            F.count(F.lit(1)).alias("n"),
            F.sum(Y).alias("sum_y"),
            F.sum(Y * Y).alias("sum_yy"),
            F.sum(Xn).alias("sum_xn"),
            F.sum(Xn * Xn).alias("sum_xnxn"),
            F.sum(Y * Xn).alias("sum_yxn"),
            F.sum(Xs).alias("sum_xs"),
            F.sum(Xs * Xs).alias("sum_xsxs"),
            F.sum(Y * Xs).alias("sum_yxs"),
            F.sum(F.col("exposure").cast("double")).alias("sum_exposure"),
        )
        .orderBy("treatment")
    )


def _data_quality(df, spark):
    """Checks that decide whether the rest of the pipeline is even meaningful."""
    from pyspark.sql import functions as F

    checks = df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("treatment").isin(0, 1), 0).otherwise(1)).alias("bad_treatment"),
        F.sum(F.when(F.col("conversion").isin(0, 1), 0).otherwise(1)).alias("bad_conversion"),
        F.sum(F.when(F.col("visit").isin(0, 1), 0).otherwise(1)).alias("bad_visit"),
        F.sum(F.when(F.col("exposure").isin(0, 1), 0).otherwise(1)).alias("bad_exposure"),
        F.sum(F.when(F.col("exposure") == 1, 1).otherwise(0)).alias("exposed"),
        # A converter who never visited would mean the two outcome columns are
        # not nested the way the dataset docs claim.
        F.sum(F.when((F.col("conversion") == 1) & (F.col("visit") == 0), 1).otherwise(0))
         .alias("converted_without_visit"),
    ).collect()[0].asDict()

    # There is no user id in this file, so "one row per user" cannot be verified
    # by key. The closest available check is exact duplicate rows across all 16
    # columns. We report the count rather than dropping: with 12 float features
    # some collisions are expected, and deleting rows would silently move the
    # split ratio the SRM check is about to test.
    distinct = df.distinct().count()
    checks["distinct_rows"] = int(distinct)
    checks["exact_duplicate_rows"] = int(checks["rows"] - distinct)
    return {k: int(v) for k, v in checks.items()}


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    if verbose:
        print(f"  spark {spark.version} on {config.SPARK_MASTER}, "
              f"java home {os.environ['JAVA_HOME'].split('/')[-3]}")

    df = _read_raw(spark).cache()

    quality = _data_quality(df, spark)
    if verbose:
        print(f"  rows                     {quality['rows']:,}")
        print(f"  distinct rows            {quality['distinct_rows']:,} "
              f"({quality['exact_duplicate_rows']:,} exact duplicates)")
        print(f"  exposed                  {quality['exposed']:,}")
        print(f"  converted without visit  {quality['converted_without_visit']:,}")

    bad = {k: v for k, v in quality.items() if k.startswith("bad_") and v}
    if bad:
        raise ValueError(f"Non-binary values in supposedly binary columns: {bad}")

    model = _fit_covariate_model(df)
    if verbose:
        print(f"  covariate model          OLS on {len(config.FEATURES)} features, "
              f"{model['fit_rows']:,} control rows, R2={model['fit_r2']:.4f}")

    df = df.withColumn("f_index", _covariate_expr(model))
    cells = _aggregate(df).toPandas()

    config.AGG_DIR.mkdir(parents=True, exist_ok=True)
    if config.CELL_PARQUET.exists():
        shutil.rmtree(config.CELL_PARQUET, ignore_errors=True)
    cells.to_parquet(config.CELL_PARQUET, index=False)

    arms = {int(r.treatment): r for r in cells.itertuples()}
    n_t, n_c = int(arms[1].n), int(arms[0].n)
    total = n_t + n_c

    out = {
        "schema": config.RAW_COLUMNS,
        "source_file": str(config.RAW_CSV if config.RAW_CSV.exists() else config.RAW_GZ),
        "quality": quality,
        "covariate_model": model,
        "arms": {
            "treatment": {
                "n": n_t,
                "conversions": int(arms[1].sum_y),
                "visits": int(arms[1].sum_xn),
                "exposed": int(arms[1].sum_exposure),
                "conversion_rate": float(arms[1].sum_y / n_t),
            },
            "control": {
                "n": n_c,
                "conversions": int(arms[0].sum_y),
                "visits": int(arms[0].sum_xn),
                "exposed": int(arms[0].sum_exposure),
                "conversion_rate": float(arms[0].sum_y / n_c),
            },
        },
        "observed_treatment_share": n_t / total,
        "observed_ratio_label": f"{round(100 * n_t / total)}/{round(100 * n_c / total)}",
        "total_rows": total,
        "cells_parquet": str(config.CELL_PARQUET),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    config.INGEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.INGEST_JSON.write_text(json.dumps(out, indent=2))

    if verbose:
        t, c = out["arms"]["treatment"], out["arms"]["control"]
        print()
        print(f"  {'arm':<10}{'n':>14}{'conversions':>14}{'rate':>10}{'share':>9}")
        print(f"  {'treatment':<10}{t['n']:>14,}{t['conversions']:>14,}"
              f"{t['conversion_rate']:>10.4%}{n_t/total:>9.2%}")
        print(f"  {'control':<10}{c['n']:>14,}{c['conversions']:>14,}"
              f"{c['conversion_rate']:>10.4%}{n_c/total:>9.2%}")
        print(f"\n  observed split           {out['observed_ratio_label']}")
        print(f"  wrote                    {config.CELL_PARQUET.name}, "
              f"{config.INGEST_JSON.name}  ({out['elapsed_seconds']}s)")

    spark.stop()
    return out


def load_cells() -> pd.DataFrame:
    """Read back the Spark aggregate for the stats stages."""
    if not config.CELL_PARQUET.exists():
        raise FileNotFoundError(
            f"No aggregate at {config.CELL_PARQUET}. Run stage 1 first."
        )
    return pd.read_parquet(config.CELL_PARQUET)


if __name__ == "__main__":
    run()
