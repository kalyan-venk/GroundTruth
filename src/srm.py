"""Stage 2 - Sample Ratio Mismatch. The guardrail that runs before anything else.

SRM asks one question: did users land in the arms in the proportion the
experiment was designed to produce? If they did not, something upstream of the
metric is broken (a bot filter that fires on one arm, a redirect that drops
users, a logging join that loses rows) and every effect estimate downstream
inherits that corruption. Microsoft's ExP team reports that a large fraction of
experiments flagged for SRM turn out to have results that reverse once fixed.

The check itself is a chi-square goodness-of-fit test on two counts. The
subtlety is not the arithmetic, it is what you test *against*.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from scipy import stats

from src import config


@dataclass
class SRMResult:
    n_treatment: int
    n_control: int
    n_total: int
    observed_treatment_share: float
    expected_treatment_share: float
    expected_treatment_count: float
    expected_control_count: float
    chi2: float
    p_value: float
    dof: int
    delta_users: float
    passed: bool
    verdict: str
    threshold: float
    binomial_sd_users: float
    z_score: float
    closeness_p: float
    underdispersed: bool
    arm_ratio: float
    nearest_simple_ratio: str
    ratio_looks_constructed: bool
    dispersion_note: str

    def as_dict(self) -> dict:
        return asdict(self)


def check(
    n_treatment: int,
    n_control: int,
    expected_treatment_share: float = config.DESIGN_TREATMENT_SHARE,
    threshold: float = 0.001,
    underdispersion_threshold: float = 0.01,
) -> SRMResult:
    """Chi-square goodness-of-fit on the arm counts.

    The arithmetic is three lines. The choices around it are the part worth
    defending.

    *What we test against.* The naive version of this check tests every
    experiment against 50/50. That is wrong for any experiment with a
    deliberately unequal allocation, and this one is deliberately 85/15 --
    Criteo downsampled the control arm because control users are the expensive
    ones (they forgo revenue). Testing an 85/15 design against 50/50 produces a
    catastrophic p-value and a completely false alarm. The expected ratio must
    come from the experiment's *design document*, not from a default.

    *The threshold.* SRM is conventionally gated at p < 0.0005 to 0.001 rather
    than the usual 0.05. That looks arbitrary until you notice the asymmetry in
    the costs: this check runs on every experiment you ever ship, so at 0.05
    one in twenty healthy experiments gets blocked for no reason, and the
    engineer's time is spent on false alarms instead of real breakage. A real
    SRM is a systems bug, and systems bugs produce absurd p-values: 1e-12,
    not 0.04. Moving the bar to 0.001 costs almost nothing in detection power
    and removes nearly all of the noise. We use 0.001.
    """
    if not 0.0 < expected_treatment_share < 1.0:
        raise ValueError("expected_treatment_share must be strictly between 0 and 1")

    n_total = n_treatment + n_control
    expected_t = n_total * expected_treatment_share
    expected_c = n_total * (1.0 - expected_treatment_share)

    chi2, p_value = stats.chisquare(
        f_obs=[n_treatment, n_control],
        f_exp=[expected_t, expected_c],
    )
    chi2, p_value = float(chi2), float(p_value)

    observed_share = n_treatment / n_total
    passed = p_value >= threshold

    # A chi-square p-value is uniform under the null, so p = 0.999 is exactly
    # as improbable as p = 0.001. Only the low tail means "broken assignment",
    # but the high tail means something too: the split is closer to the design
    # than random assignment could plausibly produce.
    #
    # Under Bernoulli(p) assignment the treatment count has sd sqrt(n*p*(1-p)).
    # If the observed count sits a small fraction of one sd from expectation,
    # the allocation was probably not drawn at random at all - it was either
    # stratified, quota-filled, or constructed after the fact by downsampling
    # an arm to hit a round number. This is Fisher's objection to Mendel's pea
    # counts, and it matters here for a practical reason: if the ratio is a
    # construction artifact, a passing SRM check says nothing about whether the
    # real assignment mechanism was sound.
    binomial_sd = (n_total * expected_treatment_share * (1.0 - expected_treatment_share)) ** 0.5
    z = (n_treatment - expected_t) / binomial_sd if binomial_sd > 0 else 0.0

    # P(|Z| < |z_observed|) under the null. This is the actual probability that
    # random assignment lands at least this close to the design, so it is a
    # p-value for the upper tail and can be gated at a stated level like any
    # other. An earlier version hard-coded a threshold of |z| < 0.01 with no
    # basis, and then printed P(|Z| < 0.01) = 0.80% as though it described the
    # observed z - quoting the probability of the threshold rather than of the
    # data, which is 7.4x too large here.
    closeness_p = float(2.0 * stats.norm.cdf(abs(z)) - 1.0)
    underdispersed = closeness_p < underdispersion_threshold

    # A second, sharper piece of evidence the z-score cannot give you.
    #
    # If an arm was downsampled to hit a round target, the arm ratio will sit
    # on a simple fraction far more precisely than sampling noise permits. So
    # search small integer ratios and report the closest. On this data
    # n_t/n_c = 5.66667239 against 17/3 = 5.66666667 - agreement to seven
    # significant figures, which no randomiser produces by accident.
    ratio = n_treatment / n_control if n_control else float("inf")
    best = min(
        ((abs(ratio - a / b), a, b) for b in range(1, 21) for a in range(1, 101)),
        key=lambda t: t[0],
    )
    ratio_gap, ratio_a, ratio_b = best
    ratio_is_exact = ratio > 0 and ratio_gap / ratio < 1e-5

    if underdispersed:
        dispersion_note = (
            f"Under-dispersed. The arms sit {abs(z):.4f} sd from the designed "
            f"split (1 sd = {binomial_sd:,.0f} users). Independent assignment "
            f"lands at least this close {closeness_p:.3%} of the time. "
            f"The split looks constructed rather than randomised, so this PASS "
            f"shows the guardrail runs - it does not validate the upstream "
            f"randomiser."
        )
        if ratio_is_exact:
            dispersion_note += (
                f" Corroborated by the arm ratio itself: {ratio:.8f} against "
                f"{ratio_a}/{ratio_b} = {ratio_a / ratio_b:.8f}, agreeing to "
                f"roughly seven significant figures."
            )
    else:
        dispersion_note = (
            f"Dispersion normal. The arms sit {abs(z):.3f} sd from the designed "
            f"split (1 sd = {binomial_sd:,.0f} users), consistent with "
            f"independent random assignment."
        )

    if passed:
        verdict = (
            f"PASS - observed split {observed_share:.4%} treatment is consistent "
            f"with the designed {expected_treatment_share:.2%} "
            f"(chi2={chi2:.3f}, p={p_value:.4f})."
        )
    else:
        verdict = (
            f"FAIL - observed split {observed_share:.4%} treatment is NOT "
            f"consistent with the designed {expected_treatment_share:.2%} "
            f"(chi2={chi2:.1f}, p={p_value:.3e}). The arms are off by "
            f"{abs(n_treatment - expected_t):,.0f} users. Downstream effect "
            f"estimates are untrustworthy until the assignment or logging "
            f"pipeline is fixed."
        )

    return SRMResult(
        n_treatment=int(n_treatment),
        n_control=int(n_control),
        n_total=int(n_total),
        observed_treatment_share=observed_share,
        expected_treatment_share=float(expected_treatment_share),
        expected_treatment_count=float(expected_t),
        expected_control_count=float(expected_c),
        chi2=chi2,
        p_value=p_value,
        dof=1,
        delta_users=float(n_treatment - expected_t),
        passed=passed,
        verdict=verdict,
        threshold=float(threshold),
        binomial_sd_users=float(binomial_sd),
        z_score=float(z),
        closeness_p=closeness_p,
        underdispersed=bool(underdispersed),
        arm_ratio=float(ratio),
        nearest_simple_ratio=f"{ratio_a}/{ratio_b}",
        ratio_looks_constructed=bool(ratio_is_exact),
        dispersion_note=dispersion_note,
    )


def sensitivity(n_treatment: int, n_control: int, shares=None) -> list[dict]:
    """What the same counts look like tested against other assumed designs.

    The point of this table is to make the dependence on the design ratio
    impossible to miss. At 13.9M rows the test has enough power to reject a
    design assumption that is off by a fraction of a percent, so the difference
    between "assumed 85%" and "assumed 84%" is the difference between a clean
    pass and a p-value with a double-digit negative exponent.
    """
    if shares is None:
        shares = [0.50, 0.80, 0.84, 0.845, 0.85, 0.855, 0.86, 0.90]
    rows = []
    for s in shares:
        r = check(n_treatment, n_control, expected_treatment_share=s)
        rows.append({
            "assumed_treatment_share": s,
            "chi2": r.chi2,
            "p_value": r.p_value,
            "passed": r.passed,
        })
    return rows


def report(result: SRMResult, sens: list[dict] | None = None) -> None:
    print(f"  designed split           {result.expected_treatment_share:.2%} treatment")
    print(f"  observed split           {result.observed_treatment_share:.4%} treatment "
          f"({result.n_treatment:,} / {result.n_control:,})")
    print(f"  expected counts          {result.expected_treatment_count:,.0f} / "
          f"{result.expected_control_count:,.0f}")
    print(f"  delta                    {result.delta_users:+,.0f} users in treatment")
    print(f"  chi2 (dof={result.dof})            {result.chi2:.4f}")
    print(f"  p-value                  {result.p_value:.6f}   "
          f"(gate at p < {result.threshold})")
    print(f"  verdict                  {'PASS' if result.passed else 'FAIL'}")
    print(f"  1 sd of binomial noise   {result.binomial_sd_users:,.0f} users")
    print(f"  observed deviation       {result.z_score:+.4f} sd")
    print(f"  P(landing this close)    {result.closeness_p:.4%}")
    if result.ratio_looks_constructed:
        a, b = (int(v) for v in result.nearest_simple_ratio.split("/"))
        print(f"  arm ratio                {result.arm_ratio:.8f}  vs  "
              f"{result.nearest_simple_ratio} = {a / b:.8f}   <-- constructed, "
              f"not randomised")
    if result.underdispersed:
        print("  ! under-dispersed        the split is closer to design than "
              "chance allows;")
        print("                           see dispersion_note in summary.json")

    if sens:
        print("\n  if the designed share were assumed to be:")
        print(f"    {'assumed':>9}{'chi2':>16}{'p':>14}   result")
        for row in sens:
            mark = "pass" if row["passed"] else "FAIL"
            print(f"    {row['assumed_treatment_share']:>8.1%}"
                  f"{row['chi2']:>16,.1f}{row['p_value']:>14.3e}   {mark}")


if __name__ == "__main__":
    import json

    data = json.loads(config.INGEST_JSON.read_text())
    res = check(data["arms"]["treatment"]["n"], data["arms"]["control"]["n"])
    report(res, sensitivity(res.n_treatment, res.n_control))
