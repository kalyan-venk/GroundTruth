"""Stage 4 - the treatment effect, unadjusted and CUPED-adjusted.

Everything here is computed from the per-arm sums Spark produced in stage 1. No
row-level data reaches this module, and none needs to: the two-proportion test
and CUPED both depend on the 13.9M rows only through n, sum(Y), sum(X),
sum(X^2) and sum(XY).

The stage answers three questions in order:

  1. What is the effect, with no adjustment?
  2. What is the effect after CUPED, and how much narrower is the interval?
  3. Does the choice of covariate matter? (Yes, enormously, and that is the
     most useful thing in this file.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

from src import config


# --- Moments from sufficient statistics -------------------------------------

@dataclass
class ArmMoments:
    """First and second moments for one arm, reconstructed from sums."""
    n: int
    mean_y: float
    var_y: float
    mean_x: float
    var_x: float
    cov_yx: float

    @property
    def corr(self) -> float:
        d = math.sqrt(self.var_y * self.var_x)
        return self.cov_yx / d if d > 0 else 0.0


def moments(n, sum_y, sum_yy, sum_x, sum_xx, sum_yx) -> ArmMoments:
    """Rebuild means, variances and covariance from raw sums.

    The textbook shortcut Var = E[X^2] - E[X]^2 is numerically dangerous when
    the mean is large relative to the spread: you subtract two nearly equal
    large numbers and lose most of your significant digits. Here it is safe,
    and worth knowing why. The metric is binary with a mean near 0.003, so
    E[X^2] and E[X]^2 differ by three orders of magnitude - there is no
    cancellation. If this pipeline ever swapped in a metric like revenue per
    user, this function would need Welford's algorithm instead, computed
    streaming inside Spark.
    """
    n = float(n)
    mean_y = sum_y / n
    mean_x = sum_x / n
    var_y = (sum_yy - n * mean_y ** 2) / (n - 1.0)
    var_x = (sum_xx - n * mean_x ** 2) / (n - 1.0)
    cov_yx = (sum_yx - n * mean_y * mean_x) / (n - 1.0)
    return ArmMoments(int(n), mean_y, var_y, mean_x, var_x, cov_yx)


def pool(a: ArmMoments, b: ArmMoments) -> ArmMoments:
    """Combine two arms' moments into the pooled moments.

    Reconstructs the sums, adds them, and re-derives. Written this way rather
    than as a weighted average of variances because the weighted-average form
    silently drops the between-arm component of the variance, which is a
    genuinely easy mistake to make and a hard one to spot.
    """
    def sums(m: ArmMoments):
        n = float(m.n)
        return (n,
                m.mean_y * n,
                m.var_y * (n - 1) + n * m.mean_y ** 2,
                m.mean_x * n,
                m.var_x * (n - 1) + n * m.mean_x ** 2,
                m.cov_yx * (n - 1) + n * m.mean_y * m.mean_x)

    sa, sb = sums(a), sums(b)
    return moments(*[x + y for x, y in zip(sa, sb)])


# --- Tests ------------------------------------------------------------------

@dataclass
class TestResult:
    label: str
    mean_treatment: float
    mean_control: float
    absolute_effect: float
    se: float
    ci_low: float
    ci_high: float
    ci_width: float
    z: float
    p_value: float
    relative_lift: float
    relative_ci_low: float
    relative_ci_high: float
    var_treatment: float
    var_control: float
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _difference_test(label, mean_t, var_t, n_t, mean_c, var_c, n_c,
                     baseline_for_lift, alpha=config.ALPHA) -> TestResult:
    """Welch-style z-test on a difference of arm means.

    Welch (unpooled variances) rather than a pooled-variance test on purpose.
    The pooled version assumes the two arms have the same variance, which is
    false by construction here: the outcome is binary, so its variance is
    p(1-p) and therefore differs between arms whenever the treatment works at
    all. With arms this size the difference is small, but assuming equal
    variance to save a term you already have is a bad habit.

    The relative lift interval divides the absolute interval by the control
    baseline, which treats that baseline as fixed. That understates the true
    uncertainty slightly. With 2.1M control users the baseline's own standard
    error is about 1% of its value, so the omitted term is second-order - but
    it is an approximation and not an exact interval.
    """
    se = math.sqrt(var_t / n_t + var_c / n_c)
    diff = mean_t - mean_c
    z = diff / se if se > 0 else 0.0
    p = float(2.0 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lo, hi = diff - crit * se, diff + crit * se

    return TestResult(
        label=label,
        mean_treatment=float(mean_t),
        mean_control=float(mean_c),
        absolute_effect=float(diff),
        se=float(se),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_width=float(hi - lo),
        z=float(z),
        p_value=p,
        relative_lift=float(diff / baseline_for_lift),
        relative_ci_low=float(lo / baseline_for_lift),
        relative_ci_high=float(hi / baseline_for_lift),
        var_treatment=float(var_t),
        var_control=float(var_c),
        significant=bool(p < alpha),
    )


def unadjusted_test(mt: ArmMoments, mc: ArmMoments) -> TestResult:
    return _difference_test(
        "unadjusted",
        mt.mean_y, mt.var_y, mt.n,
        mc.mean_y, mc.var_y, mc.n,
        baseline_for_lift=mc.mean_y,
    )


# --- CUPED ------------------------------------------------------------------

@dataclass
class CupedResult:
    covariate: str
    theta: float
    corr_pooled: float
    theoretical_variance_reduction: float
    observed_variance_reduction: float
    observed_ci_width_reduction: float
    covariate_mean_treatment: float
    covariate_mean_control: float
    covariate_imbalance: float
    covariate_imbalance_p: float
    point_estimate_shift: float
    point_estimate_shift_pct: float
    covariate_is_safe: bool
    warning: str
    test: TestResult

    def as_dict(self) -> dict:
        d = asdict(self)
        d["test"] = self.test.as_dict()
        return d


def cuped(mt: ArmMoments, mc: ArmMoments, unadjusted: TestResult,
          covariate_name: str, covariate_is_pre_assignment: bool) -> CupedResult:
    """Variance reduction using a covariate correlated with the metric.

        theta   = Cov(Y, X) / Var(X)                 estimated on pooled data
        Y_cuped = Y - theta * (X - mean(X))          mean(X) is the pooled mean

    Two implementation details that are easy to get wrong.

    *theta comes from the pooled data, not per arm.* Estimating a separate
    theta in each arm lets the adjustment absorb part of the treatment effect
    itself, which biases the estimate toward zero. One theta, both arms.

    *The centring constant is the pooled mean, not the arm mean.* Centring on
    each arm's own mean would force both adjusted means to equal their
    unadjusted values and the whole adjustment would cancel out to nothing.

    Nothing here needs row-level data. Because Y_cuped is an affine function of
    Y and X, its arm moments follow directly from the arm moments of Y and X:

        mean(Y_cuped)_a = mean(Y)_a - theta * (mean(X)_a - mean(X)_pooled)
        var(Y_cuped)_a  = var(Y)_a - 2*theta*cov(Y,X)_a + theta^2 * var(X)_a

    so the whole adjustment is arithmetic on the sums Spark already produced.
    """
    pooled = pool(mt, mc)
    theta = pooled.cov_yx / pooled.var_x if pooled.var_x > 0 else 0.0
    rho = pooled.corr

    x_bar = pooled.mean_x
    mean_t = mt.mean_y - theta * (mt.mean_x - x_bar)
    mean_c = mc.mean_y - theta * (mc.mean_x - x_bar)
    var_t = mt.var_y - 2 * theta * mt.cov_yx + theta ** 2 * mt.var_x
    var_c = mc.var_y - 2 * theta * mc.cov_yx + theta ** 2 * mc.var_x

    test = _difference_test(
        f"cuped[{covariate_name}]",
        mean_t, var_t, mt.n,
        mean_c, var_c, mc.n,
        baseline_for_lift=mc.mean_y,
    )

    var_reduction = 1.0 - (test.se ** 2) / (unadjusted.se ** 2)
    ci_reduction = 1.0 - test.ci_width / unadjusted.ci_width

    # Is the covariate balanced across arms? For a genuine pre-assignment
    # covariate it must be, up to noise - randomisation guarantees it. If it is
    # not, the covariate is being moved by the treatment, which means it is not
    # pre-assignment and CUPED will shift the point estimate rather than merely
    # tighten it.
    imbalance = mt.mean_x - mc.mean_x
    se_imbalance = math.sqrt(mt.var_x / mt.n + mc.var_x / mc.n)
    z_imbalance = imbalance / se_imbalance if se_imbalance > 0 else 0.0
    p_imbalance = float(2.0 * stats.norm.sf(abs(z_imbalance)))

    shift = test.absolute_effect - unadjusted.absolute_effect
    shift_pct = shift / unadjusted.absolute_effect if unadjusted.absolute_effect else 0.0

    if not covariate_is_pre_assignment:
        warning = (
            f"INVALID. '{covariate_name}' is measured during the experiment, so "
            f"the treatment moves it: the arms differ on it by "
            f"{imbalance:+.4%} (p={p_imbalance:.2e}). Adjusting on a variable "
            f"the treatment affects removes part of the causal path from the "
            f"estimate, which is why the effect moved by "
            f"{shift_pct:+.1%} here. The tighter interval is not a free win, "
            f"it is a differently-biased estimate. Reported for contrast only."
        )
    elif p_imbalance < 0.001:
        warning = (
            f"Covariate is pre-assignment but NOT balanced across arms: the "
            f"arms differ by {imbalance:+.6f} (p={p_imbalance:.2e}), and the "
            f"adjustment moved the point estimate by {shift_pct:+.1%}. Under "
            f"clean randomisation this shift should be near zero, so the "
            f"randomisation itself is the thing in question, not the "
            f"adjustment. Because the covariate is fixed before assignment, "
            f"the adjusted estimate is the better of the two - it corrects for "
            f"the imbalance rather than inheriting it. See the balance stage "
            f"for the per-feature picture."
        )
    else:
        warning = (
            f"Valid. Covariate is pre-assignment and balanced across arms "
            f"({imbalance:+.6f}, p={p_imbalance:.3f}), and the point estimate "
            f"moved by {shift_pct:+.2%} - within the tolerance you want, since "
            f"CUPED should tighten the interval without relocating the effect."
        )

    return CupedResult(
        covariate=covariate_name,
        theta=float(theta),
        corr_pooled=float(rho),
        theoretical_variance_reduction=float(rho ** 2),
        observed_variance_reduction=float(var_reduction),
        observed_ci_width_reduction=float(ci_reduction),
        covariate_mean_treatment=float(mt.mean_x),
        covariate_mean_control=float(mc.mean_x),
        covariate_imbalance=float(imbalance),
        covariate_imbalance_p=p_imbalance,
        point_estimate_shift=float(shift),
        point_estimate_shift_pct=float(shift_pct),
        covariate_is_safe=bool(covariate_is_pre_assignment),
        warning=warning,
        test=test,
    )


# --- Robustness: Lin's estimator --------------------------------------------

def lin_estimator(mt: ArmMoments, mc: ArmMoments,
                  unadjusted: TestResult, alpha=config.ALPHA) -> dict:
    """Regression adjustment with a separate slope per arm.

    Plain CUPED fits one theta to both arms, which quietly assumes the
    covariate relates to the outcome the same way in treatment and control. If
    the treatment changes that relationship -- if the ad works better on
    exactly the users the covariate scores highly -- a single pooled theta
    misfits both arms and can make the adjusted estimate *worse* than the
    unadjusted one.

    Lin (2013) showed the fix is to fit the slope within each arm and centre on
    the pooled covariate mean. The resulting estimator is never less efficient
    than the unadjusted difference in means asymptotically, which is the
    guarantee plain regression adjustment does not carry.

    Agreement between this and pooled CUPED is the check that matters: if they
    diverge, there is a treatment-by-covariate interaction and neither single
    number is telling the whole story.
    """
    pooled = pool(mt, mc)
    x_bar = pooled.mean_x
    theta_t = mt.cov_yx / mt.var_x if mt.var_x > 0 else 0.0
    theta_c = mc.cov_yx / mc.var_x if mc.var_x > 0 else 0.0

    mean_t = mt.mean_y - theta_t * (mt.mean_x - x_bar)
    mean_c = mc.mean_y - theta_c * (mc.mean_x - x_bar)
    var_t = mt.var_y - 2 * theta_t * mt.cov_yx + theta_t ** 2 * mt.var_x
    var_c = mc.var_y - 2 * theta_c * mc.cov_yx + theta_c ** 2 * mc.var_x

    test = _difference_test("lin", mean_t, var_t, mt.n, mean_c, var_c, mc.n,
                            baseline_for_lift=mc.mean_y)
    return {
        "theta_treatment": float(theta_t),
        "theta_control": float(theta_c),
        "theta_gap": float(theta_t - theta_c),
        "variance_reduction": float(1.0 - (test.se ** 2) / (unadjusted.se ** 2)),
        "test": test.as_dict(),
        "note": (
            "Separate slope per arm (Lin 2013). Guards against a "
            "treatment-by-covariate interaction that a single pooled theta "
            "would absorb into the effect estimate."
        ),
    }


# --- Noncompliance ----------------------------------------------------------

def cace(itt_effect: float, itt_se: float, compliance_rate: float,
         alpha: float = config.ALPHA) -> dict:
    """Effect on the users the treatment actually reached.

    Only 3.6% of the treatment arm was ever exposed to an ad, and nobody in
    control was. That is one-sided noncompliance, and it makes the
    intent-to-treat effect an average over a population that is 96% untouched.
    The ITT is still the honest headline - it is what randomisation licenses,
    and it answers "what happens if we turn this campaign on".

    Dividing ITT by the compliance rate gives the Wald estimator for the effect
    among compliers. It is valid here under two assumptions: control has zero
    access to treatment (true by construction, exposure is 0 in control), and
    assignment affects the outcome only through exposure. The second is an
    assumption, not a fact, and it is the one that would break if being
    assigned to treatment changed anything else about a user's experience.
    """
    est = itt_effect / compliance_rate
    se = itt_se / compliance_rate
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "compliance_rate": float(compliance_rate),
        "cace_absolute": float(est),
        "cace_se": float(se),
        "cace_ci_low": float(est - crit * se),
        "cace_ci_high": float(est + crit * se),
        "note": (
            "Wald/IV estimate of the effect among users actually exposed. Valid "
            "under one-sided noncompliance (control exposure is exactly 0) plus "
            "the exclusion restriction. ITT remains the primary result."
        ),
    }


# --- Orchestration for this stage -------------------------------------------

def analyse(cells) -> dict:
    """cells: the per-arm sufficient-statistics frame written by stage 1."""
    rows = {int(r.treatment): r for r in cells.itertuples()}
    t, c = rows[1], rows[0]

    safe_t = moments(t.n, t.sum_y, t.sum_yy, t.sum_xs, t.sum_xsxs, t.sum_yxs)
    safe_c = moments(c.n, c.sum_y, c.sum_yy, c.sum_xs, c.sum_xsxs, c.sum_yxs)
    naive_t = moments(t.n, t.sum_y, t.sum_yy, t.sum_xn, t.sum_xnxn, t.sum_yxn)
    naive_c = moments(c.n, c.sum_y, c.sum_yy, c.sum_xn, c.sum_xnxn, c.sum_yxn)

    unadj = unadjusted_test(safe_t, safe_c)
    cuped_safe = cuped(safe_t, safe_c, unadj, config.COVARIATE_SAFE, True)
    cuped_naive = cuped(naive_t, naive_c, unadj, config.COVARIATE_NAIVE, False)

    lin = lin_estimator(safe_t, safe_c, unadj)
    compliance = float(t.sum_exposure) / float(t.n)
    cace_res = cace(unadj.absolute_effect, unadj.se, compliance)

    return {
        "unadjusted": unadj.as_dict(),
        "cuped": cuped_safe.as_dict(),
        "cuped_naive_contrast": cuped_naive.as_dict(),
        "lin_robustness": lin,
        "cace": cace_res,
    }


def report(res: dict) -> None:
    u = res["unadjusted"]
    s = res["cuped"]
    n = res["cuped_naive_contrast"]

    print(f"  control rate             {u['mean_control']:.4%}")
    print(f"  treatment rate           {u['mean_treatment']:.4%}")
    print()
    print(f"  {'':<26}{'effect (pp)':>14}{'95% CI (pp)':>26}{'p':>13}")
    for tag, r in (("unadjusted", u), (f"CUPED [{s['covariate']}]", s["test"])):
        print(f"  {tag:<26}{r['absolute_effect']*100:>14.6f}"
              f"{'[' + format(r['ci_low']*100, '.6f') + ', ' + format(r['ci_high']*100, '.6f') + ']':>26}"
              f"{r['p_value']:>13.3e}")
    print()
    print(f"  relative lift            {u['relative_lift']:>+.2%}  "
          f"[{u['relative_ci_low']:+.2%}, {u['relative_ci_high']:+.2%}]  unadjusted")
    print(f"                           {s['test']['relative_lift']:>+.2%}  "
          f"[{s['test']['relative_ci_low']:+.2%}, {s['test']['relative_ci_high']:+.2%}]  CUPED")
    print()
    print(f"  CUPED covariate          {s['covariate']} (pre-assignment)")
    print(f"  theta                    {s['theta']:.6f}")
    print(f"  corr(Y, X) pooled        {s['corr_pooled']:.4f}")
    print(f"  variance reduction       {s['observed_variance_reduction']:.2%}   "
          f"(theory, rho^2: {s['theoretical_variance_reduction']:.2%})")
    print(f"  CI width reduction       {s['observed_ci_width_reduction']:.2%}   "
          f"(= 1 - sqrt(1 - variance reduction); these are different numbers)")
    print(f"  point estimate moved     {s['point_estimate_shift_pct']:+.3%}   "
          f"(must be ~0; CUPED tightens, it does not relocate)")
    if "lin_robustness" in res:
        li = res["lin_robustness"]
        print(f"  Lin robustness check     theta per arm "
              f"{li['theta_treatment']:.4f} / {li['theta_control']:.4f}, "
              f"effect {li['test']['absolute_effect']*100:.6f} pp "
              f"({li['test']['relative_lift']:+.2%})")
    print()
    print(f"  contrast: CUPED on '{n['covariate']}' (the tempting wrong choice)")
    print(f"    variance reduction     {n['observed_variance_reduction']:.2%}")
    print(f"    point estimate moved   {n['point_estimate_shift_pct']:+.2%}  <-- "
          f"that is the bias")
    print(f"    covariate imbalance    {n['covariate_imbalance']:+.4%} between arms, "
          f"p={n['covariate_imbalance_p']:.2e}")
    print()
    cc = res["cace"]
    print(f"  compliance (exposed)     {cc['compliance_rate']:.2%} of treatment; "
          f"0% of control")
    print(f"  CACE among exposed       {cc['cace_absolute']*100:+.4f} pp  "
          f"[{cc['cace_ci_low']*100:+.4f}, {cc['cace_ci_high']*100:+.4f}]")

    import textwrap
    for tag, w in (("CUPED", s["warning"]), ("naive contrast", n["warning"])):
        print()
        print(textwrap.fill(f"{tag}: {w}", 78,
                            initial_indent="  ", subsequent_indent="  "))


if __name__ == "__main__":
    from src.ingest import load_cells

    report(analyse(load_cells()))
