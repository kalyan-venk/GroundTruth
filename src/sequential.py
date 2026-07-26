"""Stage 7 - what happens if you look at the results early.

Every fixed-horizon test in this repo assumes the sample size was fixed before
the data arrived and the result was read once, at the end. Nobody runs an
experiment that way. Dashboards refresh, stakeholders check, and someone asks
to call it early because the number looks good.

That habit breaks the guarantee. A p-value under 0.05 means "under the null,
this happens 5% of the time" only if you look once. Look repeatedly and stop
the first time you see p < 0.05, and you are sampling the minimum of a random
walk against a fixed boundary -- which crosses far more often than 5%. Peek
daily for a month and the actual false-positive rate is roughly a quarter, not
a twentieth. Left running forever, a random walk crosses any fixed boundary
with probability 1.

So this stage measures how bad the inflation actually is, by simulation - no
formula needed, and the simulation is more convincing than one would be. Then
it builds the fix: an always-valid confidence sequence, which holds at every
sample size simultaneously, so you can look whenever you like and stop whenever
you like without spending anything you did not budget for.

The Criteo log has no timestamp, so a real sequential replay is impossible.
The alpha-inflation figures below are therefore simulated under this
experiment's actual arm sizes and baseline rate rather than measured on it, and
the confidence sequence is evaluated on the real final counts. Both are
labelled as such wherever they are printed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from src import config


@dataclass
class SequentialResult:
    n_peeks: list = field(default_factory=list)
    alpha_inflation: list = field(default_factory=list)
    fixed_horizon_ci: tuple = (0.0, 0.0)
    always_valid_ci: tuple = (0.0, 0.0)
    width_penalty: float = 0.0
    still_significant: bool = False
    verdict: str = ""

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["fixed_horizon_ci"] = list(self.fixed_horizon_ci)
        d["always_valid_ci"] = list(self.always_valid_ci)
        return d


def simulate_peeking(n_per_arm: int, baseline: float, peeks: int,
                     alpha: float = config.ALPHA, trials: int = 4000,
                     seed: int = 11) -> float:
    """False-positive rate when a NULL experiment is checked `peeks` times.

    Both arms are drawn from the same rate, so every rejection is a false
    positive by construction. Data accumulates in equal batches and a
    two-proportion z-test runs at each checkpoint; the run stops the first time
    it crosses. The fraction of runs that ever cross is the real alpha.

    Simulating the cumulative counts directly rather than user by user: the
    number of conversions in each batch is Binomial(batch, p), and the running
    total is the cumsum. Same process, no 14M-element arrays.
    """
    rng = np.random.default_rng(seed + peeks)
    batch = max(n_per_arm // peeks, 1)
    crit = stats.norm.ppf(1 - alpha / 2)

    inc_t = rng.binomial(batch, baseline, size=(trials, peeks))
    inc_c = rng.binomial(batch, baseline, size=(trials, peeks))
    cum_t, cum_c = np.cumsum(inc_t, axis=1), np.cumsum(inc_c, axis=1)
    n_seen = np.arange(1, peeks + 1) * batch

    p_t, p_c = cum_t / n_seen, cum_c / n_seen
    p_pool = (cum_t + cum_c) / (2 * n_seen)
    se = np.sqrt(np.clip(p_pool * (1 - p_pool) * 2 / n_seen, 1e-300, None))
    z = np.abs(p_t - p_c) / se

    return float((z > crit).any(axis=1).mean())


def always_valid_interval(mean_t, var_t, n_t, mean_c, var_c, n_c,
                          alpha: float = config.ALPHA, rho_tuning: float = None):
    """A confidence sequence: valid at every sample size at once.

    Uses the normal mixture boundary from Howard et al. (2021), which is what
    sits under the "always-valid p-values" in Optimizely's and Netflix's
    sequential frameworks. The fixed-horizon interval multiplies the standard
    error by z_{alpha/2}; this one multiplies it by

        sqrt( (n*rho + 1)/rho * log( (n*rho + 1) / alpha^2 ) ) / sqrt(n)

    where rho tunes which sample size the boundary is tightest at. The extra
    width is the price of being allowed to look whenever you want, and it
    shrinks only logarithmically as n grows -- so at 14M rows it costs little,
    while at 1,000 rows it would cost a lot.

    Set rho to the effective n by default, which puts the boundary's tightest
    point near where the experiment actually landed.
    """
    se = math.sqrt(var_t / n_t + var_c / n_c)
    diff = mean_t - mean_c
    n_eff = 1.0 / (1.0 / n_t + 1.0 / n_c)
    rho = rho_tuning if rho_tuning else n_eff

    boundary = math.sqrt((n_eff * rho + 1.0) / (n_eff * rho)
                         * math.log((n_eff * rho + 1.0) / (alpha ** 2)))
    radius = se * boundary

    return diff - radius, diff + radius, boundary


def analyse(mt, mc, alpha: float = config.ALPHA) -> SequentialResult:
    n_eff_per_arm = min(mt.n, mc.n)
    peek_grid = [1, 2, 5, 10, 20, 30, 50]
    inflation = [simulate_peeking(n_eff_per_arm, mc.mean_y, k, alpha=alpha)
                 for k in peek_grid]

    se = math.sqrt(mt.var_y / mt.n + mc.var_y / mc.n)
    diff = mt.mean_y - mc.mean_y
    crit = stats.norm.ppf(1 - alpha / 2)
    fixed = (diff - crit * se, diff + crit * se)

    lo, hi, boundary = always_valid_interval(
        mt.mean_y, mt.var_y, mt.n, mc.mean_y, mc.var_y, mc.n, alpha=alpha)

    penalty = (hi - lo) / (fixed[1] - fixed[0])
    still = lo > 0

    verdict = (
        f"Peeking 30 times at a null experiment this size rejects "
        f"{inflation[peek_grid.index(30)]:.1%} of the time against a nominal "
        f"{alpha:.0%}. The always-valid interval is {penalty:.2f}x wider than "
        f"the fixed-horizon one ({boundary:.2f} standard errors instead of "
        f"{crit:.2f}), and the effect "
        f"{'remains significant under it' if still else 'does not survive it'}. "
        f"At 14M rows that penalty is cheap; on a two-week experiment with "
        f"thousands of users it would not be."
    )

    return SequentialResult(
        n_peeks=peek_grid,
        alpha_inflation=inflation,
        fixed_horizon_ci=fixed,
        always_valid_ci=(lo, hi),
        width_penalty=penalty,
        still_significant=still,
        verdict=verdict,
    )


def run(cells) -> dict:
    from src import analyze

    rows = {int(r.treatment): r for r in cells.itertuples()}
    t, c = rows[1], rows[0]
    mt = analyze.moments(t.n, t.sum_y, t.sum_yy, t.sum_xs, t.sum_xsxs, t.sum_yxs)
    mc = analyze.moments(c.n, c.sum_y, c.sum_yy, c.sum_xs, c.sum_xsxs, c.sum_yxs)

    res = analyse(mt, mc).as_dict()
    (config.REPO_ROOT / "results" / "sequential.json").write_text(json.dumps(res, indent=2))
    return res


def report(res: dict) -> None:
    print("  false-positive rate when a NULL experiment is checked repeatedly")
    print("  (simulated at this experiment's arm sizes and baseline rate):")
    print(f"    {'peeks':>7}{'actual alpha':>16}{'inflation':>13}")
    for k, a in zip(res["n_peeks"], res["alpha_inflation"]):
        print(f"    {k:>7}{a:>16.1%}{a / config.ALPHA:>12.1f}x")
    print()
    f, av = res["fixed_horizon_ci"], res["always_valid_ci"]
    print(f"  fixed-horizon 95% CI     [{f[0]*100:+.6f}, {f[1]*100:+.6f}] pp")
    print(f"  always-valid CI          [{av[0]*100:+.6f}, {av[1]*100:+.6f}] pp")
    print(f"  width penalty            {res['width_penalty']:.2f}x")
    print(f"  survives peeking         {'yes' if res['still_significant'] else 'NO'}")

    import textwrap
    print()
    print(textwrap.fill(res["verdict"], 78, initial_indent="  ",
                        subsequent_indent="  "))


if __name__ == "__main__":
    from src.ingest import load_cells

    report(run(load_cells()))
