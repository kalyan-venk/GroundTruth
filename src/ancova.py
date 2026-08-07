"""Full 12-covariate ANCOVA, from the sufficient statistics.

The CUPED stage adjusts on one number per user: an OLS index that collapses
f0..f11 into a single predicted conversion propensity. That is convenient and,
as it turns out, cheap in information. The 12-feature fit reaches R^2 = 0.111
against the index's 0.106. But "the collapse costs almost nothing" is a claim,
and claims should be checked rather than asserted, which is what this stage
does.

Adjusting on all twelve separately is the textbook regression adjustment:

    Y = alpha + tau*T + X'beta + e

with tau the treatment effect. Lin's variant adds treatment-by-covariate
interactions, which is the same idea as fitting a separate slope vector per
arm.

Both are computable from X'X and X'y per arm, which stage 1 emits, so this
stage needs no Spark pass of its own. That is the payoff of having decided in
stage 1 to ship the cross-product matrix: a whole additional estimator becomes
arithmetic on numbers already in the parquet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class AncovaResult:
    n_covariates: int
    effect_pooled_slope: float
    effect_lin_interacted: float
    se_pooled: float
    se_lin: float
    relative_lift_pooled: float
    relative_lift_lin: float
    relative_ci_pooled: tuple
    relative_ci_lin: tuple
    adjusted_control_mean: float
    r2_control: float
    r2_index_only: float
    r2_gain_from_all_features: float
    verdict: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["relative_ci_pooled"] = list(self.relative_ci_pooled)
        d["relative_ci_lin"] = list(self.relative_ci_lin)
        return d


def _design(row, features):
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
    return n, xtx, xty, float(row["sum_yy"])


def _fit(n, xtx, xty, yty):
    beta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    ss_res = yty - 2 * beta @ xty + beta @ xtx @ beta
    ybar = xty[0] / n
    ss_tot = yty - n * ybar ** 2
    return beta, float(ss_res), (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def analyse(cells, index_r2: float | None = None) -> AncovaResult:
    features = config.FEATURES
    rows = {int(r["treatment"]): r for _, r in cells.iterrows()}

    n_t, xtx_t, xty_t, yty_t = _design(rows[1], features)
    n_c, xtx_c, xty_c, yty_c = _design(rows[0], features)

    beta_t, ssr_t, _ = _fit(n_t, xtx_t, xty_t, yty_t)
    beta_c, ssr_c, r2_c = _fit(n_c, xtx_c, xty_c, yty_c)

    # Pooled-slope ANCOVA. Stack the arms, fit one slope vector to both, and
    # read the treatment effect off the difference in arm means after both have
    # been centred on the pooled covariate mean.
    xtx_p, xty_p = xtx_t + xtx_c, xty_t + xty_c
    beta_p, ssr_p, _ = _fit(n_t + n_c, xtx_p, xty_p, yty_t + yty_c)

    xbar_pooled = xtx_p[0, 1:] / (n_t + n_c)
    xbar_t = xtx_t[0, 1:] / n_t
    xbar_c = xtx_c[0, 1:] / n_c
    ybar_t, ybar_c = xty_t[0] / n_t, xty_c[0] / n_c

    adj_t = ybar_t - beta_p[1:] @ (xbar_t - xbar_pooled)
    adj_c = ybar_c - beta_p[1:] @ (xbar_c - xbar_pooled)
    effect_pooled = adj_t - adj_c

    lin_t = ybar_t - beta_t[1:] @ (xbar_t - xbar_pooled)
    lin_c = ybar_c - beta_c[1:] @ (xbar_c - xbar_pooled)
    effect_lin = lin_t - lin_c

    # Residual variance per arm gives the standard errors. Using the residual
    # mean square rather than the raw outcome variance is the whole point of
    # adjusting: it is smaller by exactly the variance the covariates explain.
    p = len(features) + 1
    var_t = ssr_t / (n_t - p)
    var_c = ssr_c / (n_c - p)
    se_lin = float(np.sqrt(var_t / n_t + var_c / n_c))
    var_pool = ssr_p / (n_t + n_c - p - 1)
    se_pooled = float(np.sqrt(var_pool * (1 / n_t + 1 / n_c)))

    gain = r2_c - index_r2 if index_r2 is not None else float("nan")
    verdict = (
        f"Adjusting on all {len(features)} covariates gives "
        f"{effect_pooled * 100:+.6f} pp (pooled slope) and "
        f"{effect_lin * 100:+.6f} pp (Lin, per-arm slopes). The control-arm fit "
        f"reaches R^2 = {r2_c:.4f}"
    )
    if index_r2 is not None:
        verdict += (
            f" against {index_r2:.4f} for the single collapsed index, so the "
            f"collapse costs {gain:.4f} of R^2 - which is why the CUPED stage "
            f"can use one covariate without apology."
        )

    # Relative intervals on the log scale, the same Katz construction analyze.py
    # uses. The shortcut - take the absolute interval and divide both endpoints
    # by the control mean - is what the effect stage was caught doing, and it
    # came out 1.49x too narrow. It treats the baseline as a known constant when
    # control is the small arm of an 85/15 split, so the baseline's own sampling
    # error is nowhere near second-order. This stage had the identical bug: the
    # review fixed analyze.py and nobody checked here.
    #
    # Also worth knowing: an even earlier forest plot padded the point estimate
    # by +/-8% to have something to draw, which is a fabricated interval however
    # harmless the intent.
    #
    # One thing this does NOT model. adj_t and adj_c share the pooled slope, so
    # they are correlated, and the form below assumes independent arms. If that
    # covariance is positive - which is what you would expect when one fitted
    # slope moves both arms together - ignoring it errs wide, not narrow. I have
    # not verified the sign, so treat the pooled interval as approximate. The
    # Lin intervals fit per-arm slopes and do not have this problem.
    crit = 1.959964

    def rel_ci(mean_t, mean_c, var_mean_t, var_mean_c):
        if mean_t <= 0 or mean_c <= 0:
            return (float("nan"), float("nan"))
        se_log = float(np.sqrt(var_mean_t / mean_t ** 2 + var_mean_c / mean_c ** 2))
        log_ratio = float(np.log(mean_t / mean_c))
        return (float(np.exp(log_ratio - crit * se_log) - 1.0),
                float(np.exp(log_ratio + crit * se_log) - 1.0))

    return AncovaResult(
        n_covariates=len(features),
        effect_pooled_slope=float(effect_pooled),
        effect_lin_interacted=float(effect_lin),
        se_pooled=se_pooled,
        se_lin=se_lin,
        relative_lift_pooled=float(effect_pooled / adj_c) if adj_c else float("nan"),
        relative_lift_lin=float(effect_lin / lin_c) if lin_c else float("nan"),
        relative_ci_pooled=rel_ci(adj_t, adj_c, var_pool / n_t, var_pool / n_c),
        relative_ci_lin=rel_ci(lin_t, lin_c, var_t / n_t, var_c / n_c),
        adjusted_control_mean=float(adj_c),
        r2_control=r2_c,
        r2_index_only=index_r2 if index_r2 is not None else float("nan"),
        r2_gain_from_all_features=gain,
        verdict=verdict,
    )


def run(cells, index_r2=None) -> dict:
    res = analyse(cells, index_r2=index_r2).as_dict()
    (config.REPO_ROOT / "results" / "ancova.json").write_text(json.dumps(res, indent=2))
    return res


def report(res: dict) -> None:
    print(f"  covariates               {res['n_covariates']} "
          f"(vs 1 collapsed index in the CUPED stage)")
    print(f"  pooled-slope ANCOVA      {res['effect_pooled_slope']*100:+.6f} pp   "
          f"({res['relative_lift_pooled']:+.2%} relative)")
    print(f"  Lin, per-arm slopes      {res['effect_lin_interacted']*100:+.6f} pp   "
          f"({res['relative_lift_lin']:+.2%} relative)")
    print(f"  control-arm fit R2       {res['r2_control']:.4f}", end="")
    if res["r2_index_only"] == res["r2_index_only"]:
        print(f"   (single index: {res['r2_index_only']:.4f}, "
              f"collapse costs {res['r2_gain_from_all_features']:.4f})")
    else:
        print()


if __name__ == "__main__":
    from src.ingest import load_cells

    ing = json.loads(config.INGEST_JSON.read_text())
    report(run(load_cells(), index_r2=ing["covariate_model"]["fit_r2"]))
