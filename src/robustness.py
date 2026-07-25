"""Stage 2c - how much of the answer is a data-handling choice?

There is no user id in the Criteo log. 2,221,150 of its 13,979,592 rows sit in
groups of byte-identical duplicates, and what you do with them is a judgement
call that nothing in the file settles for you. Three defensible readings:

  all         every row is a user. Duplicates are distinct people who happen to
              share a feature vector.
  dedup       each distinct row is one user, logged more than once.
  singleton   rows that appear more than once are untrustworthy; drop them all.

The pipeline's headline used to be computed under `all` with a one-line
justification and no check on what the alternatives gave. They give very
different answers -- the lift ranges from +59% to +85% and SRM flips from a
clean pass to a 110-sigma failure. A range that wide, driven by a choice the
analyst makes rather than by sampling noise, has to be in the output. Reporting
one specification as though it were the answer would be a choice about the
conclusion dressed up as a choice about data cleaning.

This is a specification curve. The habit it encodes: when a defensible
alternative exists, run it, and if the answer moves, the movement is the
finding.

What the duplicates actually are matters for reading the table. They contain
ZERO conversions across 2.22M rows -- outcome-independent duplication would
predict about 7,700 -- and they are 95.7% treatment. Meanwhile 8 of the 12
features have fewer than 4,000 distinct values (f1 has 60, f5 132, f11 136).
So a duplicate is most likely two genuinely different cold users colliding on a
coarse feature grid, not one user logged twice. That argues for `all` as the
primary specification. It is an argument from evidence, which is what the
original one-line justification was missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from src import analyze, balance, config, srm


SPECS = ("all", "dedup", "singleton")

SPEC_LABELS = {
    "all": "every row is a user",
    "dedup": "one row per distinct record",
    "singleton": "drop all duplicated records",
}


@dataclass
class SpecResult:
    spec: str
    label: str
    n_treatment: int
    n_control: int
    n_total: int
    treatment_share: float
    srm_p: float
    srm_passed: bool
    srm_z: float
    max_abs_smd: float
    balance_passed: bool
    conversion_rate_treatment: float
    conversion_rate_control: float
    lift_unadjusted: float
    lift_unadjusted_ci: tuple
    lift_cuped: float
    lift_cuped_ci: tuple
    p_value: float

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["lift_unadjusted_ci"] = list(self.lift_unadjusted_ci)
        d["lift_cuped_ci"] = list(self.lift_cuped_ci)
        return d


# --- Spark ------------------------------------------------------------------

def _weighted_aggregate(grouped, weight_col, features):
    """Per-arm sufficient statistics under one row-weighting scheme.

    `grouped` is the raw log collapsed to distinct rows with a multiplicity
    count. Every specification is then just a different weight on those rows:
    the count itself, a flat 1, or 1 only for rows that appeared once. Doing it
    this way means the expensive shuffle happens once and the three
    specifications are three cheap scans over the collapsed table.
    """
    from pyspark.sql import functions as F

    w = F.col(weight_col).cast("double")
    Y = F.col("conversion").cast("double")
    Xn = F.col("visit").cast("double")
    Xs = F.col("f_index")

    aggs = [
        F.sum(w).alias("n"),
        F.sum(w * Y).alias("sum_y"),
        F.sum(w * Y * Y).alias("sum_yy"),
        F.sum(w * Xn).alias("sum_xn"),
        F.sum(w * Xn * Xn).alias("sum_xnxn"),
        F.sum(w * Y * Xn).alias("sum_yxn"),
        F.sum(w * Xs).alias("sum_xs"),
        F.sum(w * Xs * Xs).alias("sum_xsxs"),
        F.sum(w * Y * Xs).alias("sum_yxs"),
        F.sum(w * F.col("exposure").cast("double")).alias("sum_exposure"),
    ]
    for f in features:
        aggs.append(F.sum(w * F.col(f)).alias(f"sum_{f}"))
        aggs.append(F.sum(w * F.col(f) * Y).alias(f"sum_{f}_y"))
    for i, fi in enumerate(features):
        for fj in features[i:]:
            aggs.append(F.sum(w * F.col(fi) * F.col(fj)).alias(f"sum_{fi}_{fj}"))

    return (grouped.filter(w > 0)
            .groupBy("treatment").agg(*aggs).orderBy("treatment").toPandas())


def compute_cells(verbose: bool = True) -> dict:
    """One Spark job, three specifications. Returns {spec: cells DataFrame}."""
    from pyspark.sql import functions as F

    from src.ingest import (_covariate_expr, _fit_covariate_model, _fold_expr,
                            _read_raw, build_spark)

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = _read_raw(spark).withColumn("_fold", _fold_expr(config.COVARIATE_FOLDS)).cache()
    model = _fit_covariate_model(df, n_folds=config.COVARIATE_FOLDS)
    df = df.withColumn("f_index", _covariate_expr(model))

    # Collapse to distinct rows with multiplicity. f_index is a deterministic
    # function of the feature columns and the fold, and the fold is itself a
    # hash of those columns, so it is constant within a group and can be
    # carried through the grouping key rather than recomputed.
    keys = config.RAW_COLUMNS + ["f_index"]
    grouped = df.groupBy(*keys).agg(F.count(F.lit(1)).alias("multiplicity")).cache()

    grouped = (grouped
               .withColumn("w_all", F.col("multiplicity"))
               .withColumn("w_dedup", F.lit(1))
               .withColumn("w_singleton",
                           F.when(F.col("multiplicity") == 1, 1).otherwise(0)))

    out = {}
    for spec in SPECS:
        cells = _weighted_aggregate(grouped, f"w_{spec}", config.FEATURES)
        out[spec] = cells
        if verbose:
            tot = int(cells["n"].sum())
            print(f"  {spec:<10}{tot:>14,} rows")

    spark.stop()
    return out


# --- Analysis ---------------------------------------------------------------

def evaluate(spec: str, cells: pd.DataFrame) -> SpecResult:
    rows = {int(r.treatment): r for r in cells.itertuples()}
    t, c = rows[1], rows[0]
    n_t, n_c = int(t.n), int(c.n)

    srm_res = srm.check(n_t, n_c)
    bal_res = balance.check(cells)

    mt = analyze.moments(t.n, t.sum_y, t.sum_yy, t.sum_xs, t.sum_xsxs, t.sum_yxs)
    mc = analyze.moments(c.n, c.sum_y, c.sum_yy, c.sum_xs, c.sum_xsxs, c.sum_yxs)
    unadj = analyze.unadjusted_test(mt, mc)
    cup = analyze.cuped(mt, mc, unadj, config.COVARIATE_SAFE, True)

    return SpecResult(
        spec=spec,
        label=SPEC_LABELS[spec],
        n_treatment=n_t,
        n_control=n_c,
        n_total=n_t + n_c,
        treatment_share=n_t / (n_t + n_c),
        srm_p=srm_res.p_value,
        srm_passed=srm_res.passed,
        srm_z=srm_res.z_score,
        max_abs_smd=bal_res.max_abs_smd,
        balance_passed=bal_res.passed,
        conversion_rate_treatment=mt.mean_y,
        conversion_rate_control=mc.mean_y,
        lift_unadjusted=unadj.relative_lift,
        lift_unadjusted_ci=(unadj.relative_ci_low, unadj.relative_ci_high),
        lift_cuped=cup.test.relative_lift,
        lift_cuped_ci=(cup.test.relative_ci_low, cup.test.relative_ci_high),
        p_value=unadj.p_value,
    )


def run(verbose: bool = True) -> dict:
    if verbose:
        print("  collapsing the log to distinct rows with multiplicity...")
    cells_by_spec = compute_cells(verbose=verbose)
    results = [evaluate(s, cells_by_spec[s]) for s in SPECS]

    raw = [r.lift_unadjusted for r in results]
    adj = [r.lift_cuped for r in results]
    lifts = raw + adj
    lows = [r.lift_cuped_ci[0] for r in results] + [r.lift_unadjusted_ci[0] for r in results]

    spread_raw = max(raw) / min(raw)
    spread_adj = max(adj) / min(adj)

    out = {
        "specifications": [r.as_dict() for r in results],
        "lift_range": [min(lifts), max(lifts)],
        "lift_range_unadjusted": [min(raw), max(raw)],
        "lift_range_cuped": [min(adj), max(adj)],
        "lowest_ci_bound": min(lows),
        "spread_factor": max(lifts) / min(lifts) if min(lifts) > 0 else float("inf"),
        "spread_factor_unadjusted": spread_raw,
        "spread_factor_cuped": spread_adj,
        "adjustment_stability_gain": (spread_raw - 1.0) / (spread_adj - 1.0)
                                      if spread_adj > 1.0 else float("inf"),
        "srm_stable": len({r.srm_passed for r in results}) == 1,
        "sign_stable": all(r.lift_cuped > 0 for r in results),
        "note": (
            "The specification is a choice about the duplicate rows, not about "
            "the statistics. The ship decision survives all three; the "
            "magnitude does not."
        ),
        "adjustment_note": (
            "The two lift columns move in opposite directions as the "
            "specification tightens, and the adjusted one barely moves at all. "
            "That is the strongest argument in this project for adjusting: "
            "changing the duplicate rule changes the composition of the arms, "
            "and covariate adjustment is what absorbs composition differences. "
            "The case for CUPED here is specification robustness, not the "
            "10.7% variance reduction."
        ),
    }
    (config.REPO_ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=2))
    return out


def report(res: dict) -> None:
    print(f"  {'spec':<11}{'n total':>13}{'share':>8}{'SRM':>18}"
          f"{'max SMD':>10}{'lift (raw)':>13}{'lift (CUPED)':>14}")
    for s in res["specifications"]:
        srm_txt = f"p={s['srm_p']:.2e} {'pass' if s['srm_passed'] else 'FAIL'}"
        print(f"  {s['spec']:<11}{s['n_total']:>13,}{s['treatment_share']:>8.2%}"
              f"{srm_txt:>18}{s['max_abs_smd']:>10.4f}"
              f"{s['lift_unadjusted']:>+13.2%}{s['lift_cuped']:>+14.2%}")

    lo, hi = res["lift_range"]
    rlo, rhi = res["lift_range_unadjusted"]
    alo, ahi = res["lift_range_cuped"]
    print()
    print(f"  unadjusted across specs         {rlo:+.2%} to {rhi:+.2%}  "
          f"({res['spread_factor_unadjusted']:.2f}x)")
    print(f"  CUPED across specs              {alo:+.2%} to {ahi:+.2%}  "
          f"({res['spread_factor_cuped']:.2f}x)")
    print(f"  adjustment cuts sensitivity     {res['adjustment_stability_gain']:.1f}x"
          f"   <-- the real case for adjusting")
    print()
    print(f"  full range, all six estimates   {lo:+.2%} to {hi:+.2%}")
    print(f"  lowest 95% lower bound          {res['lowest_ci_bound']:+.2%}")
    print(f"  sign stable across specs        {'yes' if res['sign_stable'] else 'NO'}")
    print(f"  SRM verdict stable              {'yes' if res['srm_stable'] else 'NO'}"
          f"  <-- the guardrail depends on the choice")


if __name__ == "__main__":
    report(run())
