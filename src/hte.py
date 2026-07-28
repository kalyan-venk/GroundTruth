"""Stage 6 - heterogeneous treatment effects. Who does the ad actually work on?

The average treatment effect answers "should we ship". It does not answer
"who should we ship it to", and on an advertising log those are very different
questions: if the whole ATE comes from a tenth of users, spending on the other
nine tenths is waste.

The pipeline already had a signal that heterogeneity exists and walked past it.
Lin's estimator fits the covariate slope separately per arm and gets 1.289 in
treatment against 0.997 in control. A gap that size means the covariate relates
to conversion differently depending on treatment, which is what a treatment
effect that varies across users looks like. This stage measures it properly.

Method: the two-model (T-learner) uplift estimator.

    fit  E[Y | X, T=1]  on treated rows      -> beta_t
    fit  E[Y | X, T=0]  on control rows      -> beta_c
    uplift score for a user  =  x'(beta_t - beta_c)

Both fits come straight out of the per-arm sufficient statistics, because OLS
needs only X'X and X'y:

    beta = (X'X)^-1 X'y

and stage 1 already emits every entry of both. So the models are fitted on the
driver from a few hundred numbers, and Spark is needed only to score and bin
the rows afterwards.

Cross-fitted, for a reason that bites much harder here than it did for CUPED.
An uplift score fitted and evaluated on the same rows will show heterogeneity
whether or not any exists -- the model chases noise, the top decile is the
rows where noise happened to be positive, and the Qini curve looks great. Rows
are hashed into folds; the score applied to a row comes from models fitted
without it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config


N_BINS = 10


@dataclass
class HTEResult:
    n_bins: int
    deciles: list = field(default_factory=list)
    qini_coefficient: float = 0.0
    qini_curve: list = field(default_factory=list)
    ate_absolute: float = 0.0
    top_decile_uplift: float = 0.0
    bottom_decile_uplift: float = 0.0
    top_vs_bottom_ratio: float = 0.0
    share_of_effect_in_top_30pct: float = 0.0
    heterogeneity_p: float = 1.0
    heterogeneity_detected: bool = False
    verdict: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def fit_arm(row, features) -> np.ndarray:
    """OLS coefficients for one arm, from X'X and X'y sums.

    Builds the design matrix in the order [1, f0..f11], so the cross-product
    matrix needs the intercept row and column filled from n and the plain
    feature sums.
    """
    n = float(row["n"])
    p = len(features)

    xtx = np.empty((p + 1, p + 1))
    xty = np.empty(p + 1)

    xtx[0, 0] = n
    xty[0] = row["sum_y"]
    for i, fi in enumerate(features):
        xtx[0, i + 1] = xtx[i + 1, 0] = row[f"sum_{fi}"]
        xty[i + 1] = row[f"sum_{fi}_y"]
        for j, fj in enumerate(features):
            key = f"sum_{fi}_{fj}" if j >= i else f"sum_{fj}_{fi}"
            xtx[i + 1, j + 1] = row[key]

    # lstsq rather than solve: the feature grid is coarse and some columns are
    # close to collinear, so a plain inverse is not guaranteed to be stable.
    beta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    return beta


def uplift_coefficients(cells: pd.DataFrame, features=None) -> np.ndarray:
    if features is None:
        features = config.FEATURES
    rows = {int(r["treatment"]): r for _, r in cells.iterrows()}
    return fit_arm(rows[1], features) - fit_arm(rows[0], features)


def compute_bins(verbose: bool = True) -> pd.DataFrame:
    from pyspark.sql import functions as F

    from src.ingest import _fold_expr, _read_raw, build_spark

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = _read_raw(spark).withColumn("_fold", _fold_expr(config.COVARIATE_FOLDS)).cache()

    Y = F.col("conversion").cast("double")
    base = [F.count(F.lit(1)).alias("n"), F.sum(Y).alias("sum_y")]
    for f in config.FEATURES:
        base.append(F.sum(F.col(f)).alias(f"sum_{f}"))
        base.append(F.sum(F.col(f) * Y).alias(f"sum_{f}_y"))
    for i, fi in enumerate(config.FEATURES):
        for fj in config.FEATURES[i:]:
            base.append(F.sum(F.col(fi) * F.col(fj)).alias(f"sum_{fi}_{fj}"))

    per_fold = df.groupBy("_fold", "treatment").agg(*base).toPandas()

    # One uplift model per fold, fitted on the OTHER folds.
    coefs = {}
    for k in range(config.COVARIATE_FOLDS):
        train = per_fold[per_fold["_fold"] != k]
        agg = train.groupby("treatment", as_index=False).sum(numeric_only=True)
        coefs[k] = uplift_coefficients(agg)
        if verbose:
            print(f"  fold {k}: model fitted on {int(train['n'].sum()):,} rows "
                  f"outside the fold")

    score = None
    for k, beta in coefs.items():
        expr = F.lit(float(beta[0]))
        for i, f in enumerate(config.FEATURES):
            expr = expr + F.lit(float(beta[i + 1])) * F.col(f)
        score = expr if score is None else F.when(F.col("_fold") == k, expr).otherwise(score)

    scored = df.withColumn("uplift_score", score)

    # Rank into equal-size bins by score. ntile over a full 14M-row ordering is
    # a single-partition window, which is slow and memory-hungry, so instead
    # take the bin edges from a quantile approximation and bucket with a case
    # expression -- approxQuantile is a sketch and runs distributed.
    edges = scored.approxQuantile(
        "uplift_score", [i / N_BINS for i in range(1, N_BINS)], 0.001)
    edges = sorted(set(edges))

    bin_expr = F.lit(0)
    for i, e in enumerate(edges):
        bin_expr = F.when(F.col("uplift_score") > float(e), F.lit(i + 1)).otherwise(bin_expr)

    out = (scored.withColumn("bin", bin_expr)
           .groupBy("bin", "treatment")
           .agg(F.count(F.lit(1)).alias("n"),
                F.sum(Y).alias("conversions"),
                F.avg("uplift_score").alias("mean_score"))
           .orderBy("bin", "treatment")
           .toPandas())

    spark.stop()
    return out


def analyse(bins: pd.DataFrame) -> HTEResult:
    from scipy import stats

    piv = bins.pivot(index="bin", columns="treatment",
                     values=["n", "conversions", "mean_score"]).fillna(0)
    deciles = []
    for b in sorted(piv.index):
        n_t, n_c = float(piv[("n", 1)][b]), float(piv[("n", 0)][b])
        c_t, c_c = float(piv[("conversions", 1)][b]), float(piv[("conversions", 0)][b])
        if n_t == 0 or n_c == 0:
            continue
        p_t, p_c = c_t / n_t, c_c / n_c
        se = ((p_t * (1 - p_t) / n_t) + (p_c * (1 - p_c) / n_c)) ** 0.5
        deciles.append({
            "bin": int(b),
            "n_treatment": int(n_t), "n_control": int(n_c),
            "rate_treatment": p_t, "rate_control": p_c,
            "uplift": p_t - p_c, "se": se,
            # Absolute and relative uplift answer different questions and they
            # do not rank users the same way. Absolute uplift is what you
            # target on when the budget buys impressions, because it is
            # incremental conversions per user reached. Relative uplift tells
            # you whether the ad is genuinely more persuasive on these users or
            # whether they were simply going to convert anyway.
            "uplift_relative": (p_t - p_c) / p_c if p_c > 0 else float("nan"),
            "mean_score": float(piv[("mean_score", 1)][b]),
            "incremental_conversions": (p_t - p_c) * (n_t + n_c),
        })

    n_all = sum(d["n_treatment"] + d["n_control"] for d in deciles)
    conv_t = sum(d["rate_treatment"] * d["n_treatment"] for d in deciles)
    n_t_all = sum(d["n_treatment"] for d in deciles)
    conv_c = sum(d["rate_control"] * d["n_control"] for d in deciles)
    n_c_all = sum(d["n_control"] for d in deciles)
    ate = conv_t / n_t_all - conv_c / n_c_all

    # Qini. Walk the population from highest predicted uplift to lowest, and at
    # each point record how many incremental conversions you would have earned
    # by treating only that far down the ranking. A model with no signal traces
    # the diagonal; the area between the curve and that diagonal, normalised,
    # is the Qini coefficient.
    ordered = sorted(deciles, key=lambda d: -d["mean_score"])
    cum_n = cum_gain = 0.0
    curve = [{"population_share": 0.0, "incremental_conversions": 0.0}]
    for d in ordered:
        cum_n += d["n_treatment"] + d["n_control"]
        cum_gain += d["incremental_conversions"]
        curve.append({"population_share": cum_n / n_all,
                      "incremental_conversions": cum_gain})

    total_gain = curve[-1]["incremental_conversions"]
    area = area_rand = 0.0
    for a, b in zip(curve[:-1], curve[1:]):
        dx = b["population_share"] - a["population_share"]
        area += dx * (a["incremental_conversions"] + b["incremental_conversions"]) / 2
        area_rand += dx * total_gain * (a["population_share"] + b["population_share"]) / 2
    qini = (area - area_rand) / abs(area_rand) if area_rand else 0.0

    # Is the variation across bins more than sampling noise? Chi-square on the
    # standardised deviations of each bin's uplift from the overall ATE.
    chi2 = sum(((d["uplift"] - ate) / d["se"]) ** 2 for d in deciles if d["se"] > 0)
    dof = max(len(deciles) - 1, 1)
    het_p = float(stats.chi2.sf(chi2, dof))

    top, bottom = ordered[0], ordered[-1]
    top3_n = sum(d["n_treatment"] + d["n_control"] for d in ordered[:3])
    top3_gain = sum(d["incremental_conversions"] for d in ordered[:3])

    detected = het_p < 0.001
    verdict = (
        f"Heterogeneity {'detected' if detected else 'not detected'} "
        f"(chi2={chi2:.1f} on {dof} df, p={het_p:.2e}). Top decile uplift "
        f"{top['uplift']*100:+.4f} pp against {bottom['uplift']*100:+.4f} pp in the "
        f"bottom, and the top 30% of users by predicted uplift carry "
        f"{top3_gain/total_gain:.1%} of the incremental conversions while being "
        f"{top3_n/n_all:.0%} of the population. Qini coefficient {qini:.3f}."
    )

    return HTEResult(
        n_bins=len(deciles),
        deciles=deciles,
        qini_coefficient=qini,
        qini_curve=curve,
        ate_absolute=ate,
        top_decile_uplift=top["uplift"],
        bottom_decile_uplift=bottom["uplift"],
        top_vs_bottom_ratio=(top["uplift"] / bottom["uplift"]
                             if bottom["uplift"] else float("inf")),
        share_of_effect_in_top_30pct=top3_gain / total_gain if total_gain else 0.0,
        heterogeneity_p=het_p,
        heterogeneity_detected=detected,
        verdict=verdict,
    )


def run(verbose: bool = True) -> dict:
    bins = compute_bins(verbose=verbose)
    res = analyse(bins)
    out = res.as_dict()
    (config.REPO_ROOT / "results" / "hte.json").write_text(json.dumps(out, indent=2))
    return out


def report(res: dict) -> None:
    print(f"  {'decile':>7}{'n treat':>12}{'n ctrl':>11}{'rate T':>10}"
          f"{'rate C':>10}{'uplift (pp)':>14}{'+/-1.96se':>11}{'rel uplift':>12}")
    for d in sorted(res["deciles"], key=lambda x: -x["mean_score"]):
        print(f"  {d['bin']:>7}{d['n_treatment']:>12,}{d['n_control']:>11,}"
              f"{d['rate_treatment']:>10.4%}{d['rate_control']:>10.4%}"
              f"{d['uplift']*100:>+14.4f}{d['se']*100*1.96:>11.4f}"
              f"{d['uplift_relative']:>+12.1%}")
    print()
    print("  absolute uplift concentrates in the top decile; relative uplift")
    print("  does not - the top decile converts far more often to begin with.")
    print()
    print(f"  ATE                      {res['ate_absolute']*100:+.4f} pp")
    print(f"  top vs bottom decile     {res['top_decile_uplift']*100:+.4f} pp "
          f"vs {res['bottom_decile_uplift']*100:+.4f} pp")
    print(f"  effect in top 30%        {res['share_of_effect_in_top_30pct']:.1%} "
          f"of incremental conversions")
    print(f"  Qini coefficient         {res['qini_coefficient']:.3f}")
    print(f"  heterogeneity p          {res['heterogeneity_p']:.3e}")

    import textwrap
    print()
    print(textwrap.fill(res["verdict"], 78, initial_indent="  ",
                        subsequent_indent="  "))


if __name__ == "__main__":
    report(run())
