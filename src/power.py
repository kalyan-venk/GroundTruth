"""Stage 3 - power analysis. Design questions, answered before looking at the effect.

Power analysis belongs before the test, not after. Once you have seen the
result, "was this adequately powered?" stops being a design question and
becomes a rationalisation. The only input this stage takes from the data is the
*control* conversion rate, which is a property of the baseline population
rather than of the treatment effect.

The output is: given a baseline, an effect size worth detecting, alpha and
power, how many users per arm do you need, and did the experiment have them?
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src import config


@dataclass
class PowerResult:
    baseline_rate: float
    mde_relative: float
    mde_absolute: float
    treated_rate_target: float
    alpha: float
    power: float
    effect_size_h: float
    n_per_arm_balanced: int
    n_total_balanced: int
    n_control_required_at_actual_ratio: int
    n_treatment_required_at_actual_ratio: int
    n_total_required_at_actual_ratio: int
    allocation_ratio: float
    actual_n_treatment: int
    actual_n_control: int
    powered: bool
    achieved_power_at_mde: float
    mde_actually_detectable_relative: float
    verdict: str
    sensitivity: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _cohens_h(p1: float, p2: float) -> float:
    """Arcsine-transformed difference between two proportions.

    Standard power formulas assume the outcome variance is constant, but for a
    proportion the variance p(1-p) depends on p itself, so the raw difference
    p2 - p1 is not a stable effect size. The arcsine transform is the
    variance-stabilising transform for the binomial: 2*arcsin(sqrt(p)) has
    approximately constant variance regardless of p. Cohen's h is the
    difference of those transformed values, which is why statsmodels' power
    routines take h rather than a raw rate difference.
    """
    return float(proportion_effectsize(p2, p1))


def required_n_balanced(baseline: float, mde_relative: float,
                        alpha: float, power: float) -> tuple[int, float]:
    treated = baseline * (1.0 + mde_relative)
    h = _cohens_h(baseline, treated)
    n = NormalIndPower().solve_power(
        effect_size=h, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )
    return int(math.ceil(n)), h


def required_n_unbalanced(baseline: float, mde_relative: float, alpha: float,
                          power: float, treatment_share: float) -> tuple[int, int]:
    """Sample size when the arms are deliberately unequal.

    This is the part most power calculators skip, and it is the part that
    matters here. An 85/15 split does not cost you a little precision, it costs
    you a lot: the variance of the difference in means goes as 1/n_t + 1/n_c,
    and that sum is dominated by whichever arm is smaller. At 85/15 the small
    arm carries almost all of the noise.

    With k = n_t/n_c, the required control size is n_c = n_balanced * (1+1/k)/2
    and n_t = k * n_c, so the total comes to n_balanced * (2 + k + 1/k) / 2
    against 2 * n_balanced for a balanced design. At k = 5.67 that is 1.96x --
    an 85/15 experiment needs almost exactly twice the users of a 50/50 one to
    see the same effect. Worth knowing before you agree to a skewed allocation.
    """
    n_bal, _ = required_n_balanced(baseline, mde_relative, alpha, power)
    k = treatment_share / (1.0 - treatment_share)
    n_c = int(math.ceil(n_bal * (1.0 + 1.0 / k) / 2.0))
    n_t = int(math.ceil(k * n_c))
    return n_t, n_c


def achieved_power(baseline: float, mde_relative: float, alpha: float,
                   n_treatment: int, n_control: int) -> float:
    treated = baseline * (1.0 + mde_relative)
    h = _cohens_h(baseline, treated)
    n_eff = 2.0 / (1.0 / n_treatment + 1.0 / n_control)  # harmonic mean per arm
    return float(NormalIndPower().solve_power(
        effect_size=h, nobs1=n_eff, alpha=alpha, power=None,
        ratio=1.0, alternative="two-sided",
    ))


def smallest_detectable_effect(baseline: float, alpha: float, power: float,
                               n_treatment: int, n_control: int) -> float:
    """Invert the power calculation: what relative lift could these arms detect?

    Solved by bisection on relative lift rather than algebraically, because
    Cohen's h is not linear in the relative lift and inverting it in closed
    form is more error-prone than it is worth.
    """
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if achieved_power(baseline, mid, alpha, n_treatment, n_control) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def analyse(baseline_rate: float, n_treatment: int, n_control: int,
            mde_relative: float = config.MDE_RELATIVE,
            alpha: float = config.ALPHA,
            power: float = config.POWER,
            sensitivity_grid=None) -> PowerResult:
    if sensitivity_grid is None:
        sensitivity_grid = config.MDE_SENSITIVITY

    treatment_share = n_treatment / (n_treatment + n_control)
    n_bal, h = required_n_balanced(baseline_rate, mde_relative, alpha, power)
    n_t_req, n_c_req = required_n_unbalanced(
        baseline_rate, mde_relative, alpha, power, treatment_share
    )

    powered = n_treatment >= n_t_req and n_control >= n_c_req
    ach = achieved_power(baseline_rate, mde_relative, alpha, n_treatment, n_control)
    sde = smallest_detectable_effect(baseline_rate, alpha, power, n_treatment, n_control)

    sens = []
    for m in sensitivity_grid:
        nb, _ = required_n_balanced(baseline_rate, m, alpha, power)
        nt, nc = required_n_unbalanced(baseline_rate, m, alpha, power, treatment_share)
        sens.append({
            "mde_relative": m,
            "treated_rate_target": baseline_rate * (1 + m),
            "n_per_arm_balanced": nb,
            "n_treatment_required": nt,
            "n_control_required": nc,
            "n_total_required": nt + nc,
            "powered": bool(n_treatment >= nt and n_control >= nc),
            "achieved_power": achieved_power(baseline_rate, m, alpha,
                                             n_treatment, n_control),
        })

    if powered:
        verdict = (
            f"Adequately powered. Detecting a {mde_relative:.0%} relative lift on a "
            f"{baseline_rate:.4%} baseline at alpha={alpha}, power={power} needs "
            f"{n_c_req:,} control users at this 85/15 allocation; the experiment "
            f"has {n_control:,}. Achieved power at that MDE is {ach:.1%}. The "
            f"smallest lift these arms can detect at 80% power is "
            f"{sde:.2%} relative."
        )
    else:
        short = max(n_c_req - n_control, 0)
        verdict = (
            f"Underpowered. Detecting a {mde_relative:.0%} relative lift needs "
            f"{n_c_req:,} control users; the experiment has {n_control:,}, short by "
            f"{short:,}. Achieved power is only {ach:.1%}. These arms can detect "
            f"{sde:.2%} relative at 80% power, so anything smaller than that is "
            f"not reliably distinguishable from noise."
        )

    return PowerResult(
        baseline_rate=float(baseline_rate),
        mde_relative=float(mde_relative),
        mde_absolute=float(baseline_rate * mde_relative),
        treated_rate_target=float(baseline_rate * (1 + mde_relative)),
        alpha=float(alpha),
        power=float(power),
        effect_size_h=float(h),
        n_per_arm_balanced=n_bal,
        n_total_balanced=2 * n_bal,
        n_control_required_at_actual_ratio=n_c_req,
        n_treatment_required_at_actual_ratio=n_t_req,
        n_total_required_at_actual_ratio=n_t_req + n_c_req,
        allocation_ratio=float(treatment_share),
        actual_n_treatment=int(n_treatment),
        actual_n_control=int(n_control),
        powered=bool(powered),
        achieved_power_at_mde=float(ach),
        mde_actually_detectable_relative=float(sde),
        verdict=verdict,
        sensitivity=sens,
    )


def report(r: PowerResult) -> None:
    print(f"  baseline (control)       {r.baseline_rate:.4%}")
    print(f"  MDE                      {r.mde_relative:.0%} relative "
          f"({r.baseline_rate:.4%} -> {r.treated_rate_target:.4%}, "
          f"{r.mde_absolute * 100:.4f} pp absolute)")
    print(f"  alpha / power            {r.alpha} / {r.power:.0%}")
    print(f"  Cohen's h                {r.effect_size_h:.6f}")
    print()
    print(f"  n per arm, if balanced   {r.n_per_arm_balanced:>14,}  "
          f"(total {r.n_total_balanced:,})")
    print(f"  n required at {r.allocation_ratio:.0%}/{1-r.allocation_ratio:.0%}    "
          f"treatment {r.n_treatment_required_at_actual_ratio:,}, "
          f"control {r.n_control_required_at_actual_ratio:,}")
    print(f"                           total {r.n_total_required_at_actual_ratio:,} "
          f"({r.n_total_required_at_actual_ratio / r.n_total_balanced:.2f}x the "
          f"balanced design)")
    print(f"  actually had             treatment {r.actual_n_treatment:,}, "
          f"control {r.actual_n_control:,}")
    print()
    print(f"  achieved power at MDE    {r.achieved_power_at_mde:.2%}")
    print(f"  smallest detectable      {r.mde_actually_detectable_relative:.2%} "
          f"relative, at 80% power")
    print(f"  verdict                  {'POWERED' if r.powered else 'UNDERPOWERED'}")

    print("\n  sensitivity to the MDE choice:")
    print(f"    {'rel MDE':>8}{'target rate':>14}{'n control req':>16}"
          f"{'n total req':>14}{'power':>9}   result")
    for s in r.sensitivity:
        print(f"    {s['mde_relative']:>8.0%}{s['treated_rate_target']:>14.4%}"
              f"{s['n_control_required']:>16,}{s['n_total_required']:>14,}"
              f"{s['achieved_power']:>9.1%}   "
              f"{'ok' if s['powered'] else 'UNDERPOWERED'}")


if __name__ == "__main__":
    import json

    data = json.loads(config.INGEST_JSON.read_text())
    res = analyse(
        baseline_rate=data["arms"]["control"]["conversion_rate"],
        n_treatment=data["arms"]["treatment"]["n"],
        n_control=data["arms"]["control"]["n"],
    )
    report(res)
