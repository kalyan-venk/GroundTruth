"""Correctness tests for the statistics.

The pipeline's claim is that its numbers are defensible, so the tests check the
maths against ground truth rather than checking that the code runs. Each test
constructs data where the right answer is known by construction and asserts the
implementation recovers it.

    .venv/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import analyze, ancova, balance, config, power, srm


RNG = np.random.default_rng(20260725)


def sums_from(y: np.ndarray, x: np.ndarray):
    """Pack raw arrays into the sufficient statistics ingest would emit."""
    return dict(
        n=len(y), sum_y=y.sum(), sum_yy=(y * y).sum(),
        sum_x=x.sum(), sum_xx=(x * x).sum(), sum_yx=(y * x).sum(),
    )


# --- moments ----------------------------------------------------------------

def test_moments_match_numpy():
    y = RNG.binomial(1, 0.03, 50_000).astype(float)
    x = RNG.normal(5.0, 2.0, 50_000)
    m = analyze.moments(**sums_from(y, x))

    assert m.mean_y == pytest.approx(y.mean(), rel=1e-12)
    assert m.var_y == pytest.approx(y.var(ddof=1), rel=1e-10)
    assert m.var_x == pytest.approx(x.var(ddof=1), rel=1e-10)
    assert m.cov_yx == pytest.approx(np.cov(y, x, ddof=1)[0, 1], rel=1e-10)


def test_pool_matches_concatenation():
    """Pooling two arms' moments must equal computing moments on the union.

    This is the one that catches the classic bug: averaging the two arms'
    variances drops the between-arm term and gives a number that is too small.
    """
    ya, xa = RNG.binomial(1, 0.02, 30_000).astype(float), RNG.normal(3, 1, 30_000)
    yb, xb = RNG.binomial(1, 0.05, 12_000).astype(float), RNG.normal(6, 2, 12_000)

    pooled = analyze.pool(analyze.moments(**sums_from(ya, xa)),
                          analyze.moments(**sums_from(yb, xb)))
    direct = analyze.moments(**sums_from(np.concatenate([ya, yb]),
                                         np.concatenate([xa, xb])))

    assert pooled.mean_y == pytest.approx(direct.mean_y, rel=1e-10)
    assert pooled.var_y == pytest.approx(direct.var_y, rel=1e-9)
    assert pooled.var_x == pytest.approx(direct.var_x, rel=1e-9)
    assert pooled.cov_yx == pytest.approx(direct.cov_yx, rel=1e-9)


# --- SRM --------------------------------------------------------------------

def test_srm_passes_on_a_clean_split():
    res = srm.check(n_treatment=850_123, n_control=149_877,
                    expected_treatment_share=0.85)
    assert res.passed
    assert res.p_value > 0.001


def test_srm_catches_a_planted_mismatch():
    """Drop 2% of the treatment arm, as a broken redirect would."""
    res = srm.check(n_treatment=833_000, n_control=150_000,
                    expected_treatment_share=0.85)
    assert not res.passed
    assert res.p_value < 1e-6


def test_srm_against_the_wrong_expected_ratio_produces_a_false_alarm():
    """A healthy 85/15 experiment tested against 50/50 must fail loudly.

    Guards the choice that matters most in this stage: the expected ratio has
    to come from the design, not from a default.
    """
    healthy = srm.check(850_000, 150_000, expected_treatment_share=0.85)
    misjudged = srm.check(850_000, 150_000, expected_treatment_share=0.50)
    assert healthy.passed
    assert not misjudged.passed


def test_srm_flags_underdispersion():
    """An exactly-on-design split is suspicious, not reassuring."""
    exact = srm.check(8_500_000, 1_500_000, expected_treatment_share=0.85)
    assert exact.passed
    assert exact.underdispersed

    noisy = srm.check(8_502_400, 1_497_600, expected_treatment_share=0.85)
    assert noisy.passed
    assert not noisy.underdispersed


# --- Power ------------------------------------------------------------------

def test_required_n_round_trips_to_the_target_power():
    baseline, mde = 0.002, 0.05
    n, _ = power.required_n_balanced(baseline, mde, alpha=0.05, power=0.80)
    achieved = power.achieved_power(baseline, mde, 0.05, n, n)
    assert achieved == pytest.approx(0.80, abs=1e-4)


def test_unbalanced_allocation_costs_users():
    baseline, mde = 0.002, 0.05
    n_bal, _ = power.required_n_balanced(baseline, mde, 0.05, 0.80)
    n_t, n_c = power.required_n_unbalanced(baseline, mde, 0.05, 0.80, 0.85)
    assert (n_t + n_c) / (2 * n_bal) == pytest.approx(1.96, abs=0.02)


def test_smallest_detectable_effect_inverts_the_power_curve():
    baseline = 0.002
    n_t, n_c = 11_882_655, 2_096_937
    sde = power.smallest_detectable_effect(baseline, 0.05, 0.80, n_t, n_c)
    assert power.achieved_power(baseline, sde, 0.05, n_t, n_c) == pytest.approx(0.80, abs=1e-3)


def test_smaller_mde_needs_more_users():
    baseline = 0.002
    sizes = [power.required_n_balanced(baseline, m, 0.05, 0.80)[0]
             for m in (0.20, 0.10, 0.05, 0.02)]
    assert sizes == sorted(sizes)


# --- The unadjusted test ----------------------------------------------------

def test_unadjusted_matches_scipy_on_a_known_effect():
    n_t, n_c, p_t, p_c = 200_000, 200_000, 0.030, 0.025
    y_t = np.zeros(n_t); y_t[:int(n_t * p_t)] = 1.0
    y_c = np.zeros(n_c); y_c[:int(n_c * p_c)] = 1.0
    x_t, x_c = RNG.normal(0, 1, n_t), RNG.normal(0, 1, n_c)

    mt = analyze.moments(**sums_from(y_t, x_t))
    mc = analyze.moments(**sums_from(y_c, x_c))
    res = analyze.unadjusted_test(mt, mc)

    expected_t, expected_p = stats.ttest_ind(y_t, y_c, equal_var=False)
    assert res.z == pytest.approx(expected_t, rel=1e-4)
    assert res.p_value == pytest.approx(expected_p, rel=1e-3)
    assert res.absolute_effect == pytest.approx(p_t - p_c, abs=1e-6)


def test_relative_ci_accounts_for_baseline_uncertainty():
    """The ratio interval must not treat the control mean as a known constant.

    Regression test for a real bug. The first version divided the absolute CI
    by the control mean, which produced intervals 1.49x too narrow on the real
    data. The docstring defending it argued the baseline's own error was
    "second-order"; it is not, because control is the small arm in an 85/15
    split and therefore carries most of the variance.

    Checked against the Katz log interval computed independently here.
    """
    n_t, n_c, p_t, p_c = 11_882_655, 2_096_937, 0.0030894610674, 0.0019375880153
    y_t = np.array([1.0] * round(n_t * p_t) + [0.0] * (n_t - round(n_t * p_t)))
    y_c = np.array([1.0] * round(n_c * p_c) + [0.0] * (n_c - round(n_c * p_c)))
    x_t, x_c = np.zeros(n_t), np.zeros(n_c)

    res = analyze.unadjusted_test(analyze.moments(**sums_from(y_t, x_t)),
                                  analyze.moments(**sums_from(y_c, x_c)))

    se_log = math.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
    lo = math.exp(math.log(p_t / p_c) - 1.959964 * se_log) - 1
    hi = math.exp(math.log(p_t / p_c) + 1.959964 * se_log) - 1

    assert res.relative_ci_low == pytest.approx(lo, rel=1e-3)
    assert res.relative_ci_high == pytest.approx(hi, rel=1e-3)

    # And it must be materially wider than the naive fixed-baseline interval.
    naive_width = (res.ci_high - res.ci_low) / p_c
    assert (res.relative_ci_high - res.relative_ci_low) / naive_width > 1.3


def test_adjusted_lift_uses_the_adjusted_baseline():
    """CUPED's relative lift must divide by CUPED's own control mean.

    Regression test. Dividing an adjusted numerator by the unadjusted control
    mean reported +50.76% where the consistent figure was +47.27%. It is the
    wrong denominator under CUPED's own premise -- the argument for adjusting
    is that the unadjusted control mean carries the imbalance.
    """
    mt, mc = _synthetic_experiment(rho=0.7, effect=0.01, shift=0.05)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)

    implied = res.test.mean_treatment / res.test.mean_control - 1.0
    assert res.test.relative_lift == pytest.approx(implied, rel=1e-9)

    wrong = res.test.absolute_effect / mc.mean_y
    assert res.test.relative_lift != pytest.approx(wrong, rel=1e-6)


def _ancova_cells(n_t=60_000, n_c=40_000, base=0.03, effect=0.004):
    """Two arms of 12-covariate sufficient statistics, the shape ingest emits."""
    rows = []
    for treat, n in ((1, n_t), (0, n_c)):
        x = RNG.normal(0.0, 1.0, (n, len(config.FEATURES)))
        p = base + effect * treat + 0.004 * x[:, 0]
        y = RNG.binomial(1, np.clip(p, 1e-6, 1 - 1e-6)).astype(float)

        row = {"treatment": treat, "n": n, "sum_y": y.sum(), "sum_yy": (y * y).sum()}
        for i, fi in enumerate(config.FEATURES):
            row[f"sum_{fi}"] = x[:, i].sum()
            row[f"sum_{fi}_y"] = (x[:, i] * y).sum()
            for j, fj in enumerate(config.FEATURES):
                if j >= i:
                    row[f"sum_{fi}_{fj}"] = (x[:, i] * x[:, j]).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def test_ancova_relative_ci_is_not_the_fixed_baseline_shortcut():
    """The same bug as the test above, in the stage nobody re-checked.

    The review caught analyze.py dividing the absolute interval by the control
    mean. ancova.py was doing it too and was not part of that fix, so the forest
    plot shipped a pooled interval 1.37x too narrow on the real data. Worth a
    test because the failure mode here is a fix landing in one of two call
    sites, not the maths being hard.
    """
    res = ancova.analyse(_ancova_cells())
    lift = res.relative_lift_pooled
    lo, hi = res.relative_ci_pooled
    assert lo < lift < hi

    # The tell is the shape, not the width. A ratio interval built on the log
    # scale is symmetric in log space; the fixed-baseline shortcut divides an
    # already-symmetric absolute interval, so it stays symmetric in linear
    # space. Checking the shape catches the bug on any data, where a width
    # threshold only catches it when the arms happen to be lopsided enough.
    assert math.log1p(hi) - math.log1p(lift) == pytest.approx(
        math.log1p(lift) - math.log1p(lo), rel=1e-9)
    assert (hi - lift) != pytest.approx(lift - lo, rel=1e-6)


def test_variance_reduction_decomposes_by_arm_weight():
    """Observed reduction must equal the Var(diff)-weighted mix of arm reductions.

    This is the identity that explains why the observed 10.70% on the real data
    sits below the pooled rho^2 of 11.81%: what gets reduced is
    var_t/n_t + var_c/n_c, so each arm's reduction counts in proportion to its
    share of that sum, and in an 85/15 split control carries 78% of it.
    """
    mt, mc = _synthetic_experiment(n_t=600_000, n_c=150_000, rho=0.6)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)

    assert res.achievable_variance_reduction == pytest.approx(
        res.observed_variance_reduction, abs=1e-9)
    assert res.control_share_of_variance > 0.5   # small arm dominates


def test_no_effect_gives_a_boring_p_value():
    y_t = RNG.binomial(1, 0.02, 100_000).astype(float)
    y_c = RNG.binomial(1, 0.02, 100_000).astype(float)
    x_t, x_c = RNG.normal(0, 1, 100_000), RNG.normal(0, 1, 100_000)
    res = analyze.unadjusted_test(analyze.moments(**sums_from(y_t, x_t)),
                                  analyze.moments(**sums_from(y_c, x_c)))
    assert res.p_value > 0.01


# --- CUPED ------------------------------------------------------------------

def _synthetic_experiment(n_t=400_000, n_c=400_000, rho=0.6, effect=0.01, shift=0.0):
    """Build an experiment where the truth is known by construction.

    X is a pre-assignment covariate. Y is X-driven noise plus a fixed additive
    treatment effect. `shift` deliberately imbalances X between the arms so the
    imbalance path can be tested too.
    """
    x_t = RNG.normal(shift, 1.0, n_t)
    x_c = RNG.normal(0.0, 1.0, n_c)
    noise = math.sqrt(1 - rho ** 2)
    y_t = rho * x_t + noise * RNG.normal(0, 1, n_t) + effect
    y_c = rho * x_c + noise * RNG.normal(0, 1, n_c)
    return (analyze.moments(**sums_from(y_t, x_t)),
            analyze.moments(**sums_from(y_c, x_c)))


def test_cuped_reduces_variance_by_about_rho_squared():
    mt, mc = _synthetic_experiment(rho=0.6)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)

    assert res.observed_variance_reduction == pytest.approx(0.36, abs=0.02)
    assert res.observed_variance_reduction == pytest.approx(
        res.theoretical_variance_reduction, abs=0.01)


def test_cuped_does_not_move_the_point_estimate_when_arms_are_balanced():
    """The correctness check from the kickstart, stated so it is actually testable.

    "CUPED must not change the point estimate meaningfully" needs a yardstick.
    Percentage of the effect is the wrong one: even under perfect
    randomisation the arms differ on X by sampling noise, CUPED corrects that
    difference, and the correction is theta * se(X_t - X_c) regardless of how
    big the effect is. A small effect therefore moves a long way in percentage
    terms for entirely innocent reasons -- an earlier version of this test
    asserted <5% and failed at 16% on data that was behaving perfectly.

    The right yardstick is the shift against that chance benchmark. Roughly 1x
    is a healthy experiment. On the real Criteo data it comes out around 12x,
    which is how we know that shift is systematic rather than noise.
    """
    mt, mc = _synthetic_experiment(rho=0.7, effect=0.01, shift=0.0)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)

    assert res.shift_vs_chance_multiple < 4.0
    assert res.test.ci_width < unadj.ci_width


def test_shift_vs_chance_separates_real_imbalance_from_noise():
    balanced_mt, balanced_mc = _synthetic_experiment(rho=0.7, effect=0.01, shift=0.0)
    skewed_mt, skewed_mc = _synthetic_experiment(rho=0.7, effect=0.01, shift=0.05)

    clean = analyze.cuped(balanced_mt, balanced_mc,
                          analyze.unadjusted_test(balanced_mt, balanced_mc),
                          "x", covariate_is_pre_assignment=True)
    dirty = analyze.cuped(skewed_mt, skewed_mc,
                          analyze.unadjusted_test(skewed_mt, skewed_mc),
                          "x", covariate_is_pre_assignment=True)

    assert clean.shift_vs_chance_multiple < 4.0
    assert dirty.shift_vs_chance_multiple > 10.0


def test_cuped_recovers_the_true_effect_despite_covariate_imbalance():
    """With X imbalanced, the unadjusted estimate is biased and CUPED fixes it.

    This is the mechanism behind the -14.6% shift on the real data, isolated:
    plant a 0.05 sd imbalance in X, and the unadjusted difference picks up
    rho * 0.05 = 0.035 of spurious effect on top of the true 0.01. CUPED should
    subtract it back out.
    """
    true_effect, shift, rho = 0.01, 0.05, 0.7
    mt, mc = _synthetic_experiment(n_t=800_000, n_c=800_000,
                                   rho=rho, effect=true_effect, shift=shift)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)

    assert unadj.absolute_effect == pytest.approx(true_effect + rho * shift, abs=0.005)
    assert res.test.absolute_effect == pytest.approx(true_effect, abs=0.005)
    assert abs(res.test.absolute_effect - true_effect) < abs(unadj.absolute_effect - true_effect)


def test_cuped_is_flagged_when_the_covariate_is_post_treatment():
    mt, mc = _synthetic_experiment(shift=0.3)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "visit", covariate_is_pre_assignment=False)
    assert not res.covariate_is_safe
    assert res.warning.startswith("INVALID")


def test_useless_covariate_buys_nothing():
    mt, mc = _synthetic_experiment(rho=0.0)
    unadj = analyze.unadjusted_test(mt, mc)
    res = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)
    assert res.observed_variance_reduction == pytest.approx(0.0, abs=0.01)


def test_lin_matches_cuped_when_slopes_are_equal():
    mt, mc = _synthetic_experiment(rho=0.6, effect=0.01)
    unadj = analyze.unadjusted_test(mt, mc)
    cup = analyze.cuped(mt, mc, unadj, "x", covariate_is_pre_assignment=True)
    lin = analyze.lin_estimator(mt, mc, unadj)
    assert lin["test"]["absolute_effect"] == pytest.approx(
        cup.test.absolute_effect, abs=0.002)


# --- Balance ----------------------------------------------------------------

def _balance_frame(mean_t, mean_c, n_t=300_000, n_c=300_000, features=("a", "b")):
    import pandas as pd

    rows = []
    for arm, means, n in ((1, mean_t, n_t), (0, mean_c, n_c)):
        data = {f: RNG.normal(m, 1.0, n) for f, m in zip(features, means)}
        row = {"treatment": arm, "n": n}
        for f in features:
            row[f"sum_{f}"] = data[f].sum()
        for i, fi in enumerate(features):
            for fj in features[i:]:
                row[f"sum_{fi}_{fj}"] = (data[fi] * data[fj]).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def test_balance_passes_on_identical_arms():
    frame = _balance_frame((0.0, 0.0), (0.0, 0.0))
    res = balance.check(frame, features=["a", "b"])
    assert res.passed
    assert res.max_abs_smd < 0.02


def test_balance_catches_a_planted_shift():
    frame = _balance_frame((0.30, 0.0), (0.0, 0.0))
    res = balance.check(frame, features=["a", "b"])
    assert not res.passed
    assert res.max_abs_smd_feature == "a"
    assert res.max_abs_smd == pytest.approx(0.30, abs=0.02)
    assert res.hotelling_p < 1e-6


def test_hotelling_catches_a_shift_no_single_feature_shows():
    """The case univariate checks miss.

    Two features each shifted by 0.02 sd -- individually well under any
    threshold -- but shifted together along a correlated direction. The joint
    test should see what twelve separate t-tests would wave through.
    """
    frame = _balance_frame((0.02, 0.02), (0.0, 0.0), n_t=2_000_000, n_c=2_000_000)
    res = balance.check(frame, features=["a", "b"])
    assert res.max_abs_smd < 0.10          # every feature individually "fine"
    assert res.hotelling_p < 1e-6          # jointly, not fine
