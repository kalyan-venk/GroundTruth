"""Stage 2b - covariate balance. The guardrail SRM cannot give you.

SRM asks whether the arms have the right *number* of users. Balance asks
whether they have the right *kind*. Those are different failures and a broken
experiment can pass one while failing the other, which is exactly what happens
in this dataset: the split is 85/15 to within 2 users out of 14 million, and
the arms still differ measurably on the pre-assignment features.

Randomisation is what licenses the causal claim, and what randomisation
actually guarantees is that arms are comparable on *everything*, observed and
unobserved. Observed imbalance is the only symptom you can see. If the observed
features are off, the sensible inference is that unobserved ones are too.

Hence two tests:

  - Standardised mean difference, per feature. Scale-free measure of how far
    apart the arms are. Convention: |SMD| < 0.1 is negligible. This does not
    care about sample size, which is the point.
  - Hotelling's T^2, all 12 features jointly. The multivariate version of a
    two-sample t-test. Catches the case where every individual feature looks
    fine but the joint distribution is shifted along some diagonal.

Both come from the per-arm sums and cross-products Spark produced in stage 1,
so neither needs row-level data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import stats

from src import config

SMD_NEGLIGIBLE = 0.10


@dataclass
class BalanceResult:
    n_treatment: int
    n_control: int
    n_features: int
    max_abs_smd: float
    max_abs_smd_feature: str
    n_features_above_threshold: int
    smd_threshold: float
    hotelling_t2: float
    hotelling_f: float
    hotelling_df1: int
    hotelling_df2: int
    hotelling_p: float
    passed: bool
    verdict: str
    per_feature: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _arm_moments(row, features):
    n = float(row["n"])
    mean = np.array([row[f"sum_{f}"] / n for f in features])

    cov = np.empty((len(features), len(features)))
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            key = f"sum_{fi}_{fj}" if j >= i else f"sum_{fj}_{fi}"
            cov[i, j] = (row[key] - n * mean[i] * mean[j]) / (n - 1.0)
    return n, mean, cov


def check(cells, features=None, smd_threshold: float = SMD_NEGLIGIBLE) -> BalanceResult:
    if features is None:
        features = config.FEATURES

    rows = {int(r["treatment"]): r for _, r in cells.iterrows()}
    n_t, mean_t, cov_t = _arm_moments(rows[1], features)
    n_c, mean_c, cov_c = _arm_moments(rows[0], features)

    # --- per-feature standardised mean differences -------------------------
    pooled_sd = np.sqrt((np.diag(cov_t) + np.diag(cov_c)) / 2.0)
    diff = mean_t - mean_c
    smd = np.divide(diff, pooled_sd, out=np.zeros_like(diff), where=pooled_sd > 0)

    se = np.sqrt(np.diag(cov_t) / n_t + np.diag(cov_c) / n_c)
    z = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
    p = 2.0 * stats.norm.sf(np.abs(z))

    per_feature = [
        {
            "feature": f,
            "mean_treatment": float(mean_t[i]),
            "mean_control": float(mean_c[i]),
            "difference": float(diff[i]),
            "smd": float(smd[i]),
            "z": float(z[i]),
            "p_value": float(p[i]),
            "above_threshold": bool(abs(smd[i]) > smd_threshold),
        }
        for i, f in enumerate(features)
    ]

    # --- Hotelling's T^2 ---------------------------------------------------
    # Two-sample multivariate test on the mean vectors. Uses the pooled
    # covariance and converts to an F statistic:
    #
    #   T^2 = (n_t*n_c/(n_t+n_c)) * d' S^-1 d
    #   F   = T^2 * (N - p - 1) / (p * (N - 2)),  df = (p, N - p - 1)
    #
    # solve() rather than an explicit inverse: forming S^-1 and multiplying is
    # both slower and less numerically stable than solving the system, and with
    # 12 correlated ad-tech features the matrix is not especially well
    # conditioned.
    p_dim = len(features)
    n_total = n_t + n_c
    s_pooled = ((n_t - 1) * cov_t + (n_c - 1) * cov_c) / (n_total - 2)

    try:
        solved = np.linalg.solve(s_pooled, diff)
        singular = False
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(s_pooled) @ diff
        singular = True

    t2 = float((n_t * n_c / n_total) * diff @ solved)
    f_stat = float(t2 * (n_total - p_dim - 1) / (p_dim * (n_total - 2)))
    df1, df2 = p_dim, int(n_total - p_dim - 1)
    p_omnibus = float(stats.f.sf(f_stat, df1, df2))

    max_i = int(np.argmax(np.abs(smd)))
    n_above = int(sum(1 for r in per_feature if r["above_threshold"]))
    passed = n_above == 0

    if passed and p_omnibus >= 0.001:
        verdict = (
            f"Balanced. No feature exceeds |SMD| = {smd_threshold}, and the "
            f"joint test does not reject (Hotelling F={f_stat:.2f}, "
            f"p={p_omnibus:.3f})."
        )
    elif passed:
        verdict = (
            f"Statistically imbalanced, practically negligible. The joint test "
            f"rejects decisively (Hotelling F={f_stat:,.1f}, p={p_omnibus:.2e}) "
            f"but the largest standardised difference is only "
            f"{abs(smd[max_i]):.4f} on {features[max_i]}, far under the {smd_threshold} "
            f"convention. At N={int(n_total):,} the test has power to detect "
            f"imbalance far smaller than anything that would matter on a normal "
            f"metric. It matters here anyway, because the treatment effect is "
            f"itself tiny - see the CUPED point-estimate shift."
        )
    else:
        verdict = (
            f"IMBALANCED. {n_above} of {p_dim} features exceed |SMD| = "
            f"{smd_threshold}, worst is {features[max_i]} at {smd[max_i]:+.4f} "
            f"(Hotelling F={f_stat:,.1f}, p={p_omnibus:.2e}). The arms are not "
            f"exchangeable and the unadjusted difference in means is not a "
            f"clean causal estimate."
        )

    if singular:
        verdict += " (Covariance matrix was singular; pseudo-inverse used.)"

    return BalanceResult(
        n_treatment=int(n_t),
        n_control=int(n_c),
        n_features=p_dim,
        max_abs_smd=float(abs(smd[max_i])),
        max_abs_smd_feature=features[max_i],
        n_features_above_threshold=n_above,
        smd_threshold=float(smd_threshold),
        hotelling_t2=t2,
        hotelling_f=f_stat,
        hotelling_df1=df1,
        hotelling_df2=df2,
        hotelling_p=p_omnibus,
        passed=passed,
        verdict=verdict,
        per_feature=per_feature,
    )


def report(r: BalanceResult, top: int = 12) -> None:
    print(f"  features tested          {r.n_features}")
    print(f"  Hotelling T2             {r.hotelling_t2:,.1f}  ->  "
          f"F({r.hotelling_df1}, {r.hotelling_df2:,}) = {r.hotelling_f:,.2f}, "
          f"p = {r.hotelling_p:.3e}")
    print(f"  largest |SMD|            {r.max_abs_smd:.4f} on {r.max_abs_smd_feature} "
          f"(negligible threshold {r.smd_threshold})")
    print(f"  features over threshold  {r.n_features_above_threshold} of {r.n_features}")
    print()
    print(f"    {'feature':<9}{'mean treat':>14}{'mean ctrl':>14}"
          f"{'SMD':>10}{'z':>10}{'p':>12}")
    ordered = sorted(r.per_feature, key=lambda d: -abs(d["smd"]))[:top]
    for row in ordered:
        flag = "  <-- over threshold" if row["above_threshold"] else ""
        print(f"    {row['feature']:<9}{row['mean_treatment']:>14.5f}"
              f"{row['mean_control']:>14.5f}{row['smd']:>10.4f}"
              f"{row['z']:>10.1f}{row['p_value']:>12.2e}{flag}")


if __name__ == "__main__":
    from src.ingest import load_cells

    res = check(load_cells())
    report(res)
    print()
    import textwrap
    print(textwrap.fill(res.verdict, 78, initial_indent="  ", subsequent_indent="  "))
