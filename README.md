# GroundTruth

An A/B test analysis pipeline on the Criteo Uplift log — 13,979,592 rows through
Spark, four statistical stages, one ship/no-ship verdict. The point of the
project is experimental rigor, not model training: validate the experiment,
size it, measure it, and be honest about what the measurement is worth.

```bash
bash scripts/get_data.sh          # 311 MB download, decompresses to 3 GB
python -m src.run                 # full pipeline, ~33 seconds
python -m src.run --skip-spark    # stats only, reuses the aggregate
python -m pytest tests/ -q        # 22 tests
```

Every number below comes out of `results/summary.json`, written by that run.

---

## Result

**Ship.** The campaign lifts conversion, and the effect is nowhere near the
noise floor.

| | effect (pp) | 95% CI | relative lift | p |
|---|---|---|---|---|
| unadjusted | +0.115187 | [+0.108451, +0.121924] | **+59.45%** | 3.2e-246 |
| CUPED-adjusted | +0.098359 | [+0.091993, +0.104725] | **+50.76%** | 2.0e-201 |
| Lin (robustness) | +0.101138 | — | +52.20% | — |

The headline is the **adjusted** number, not the raw one, and the reason is the
most interesting thing this pipeline found. More below.

| | |
|---|---|
| SRM p-value | 0.9989 (pass — and see the caveat) |
| Observed split | 85/15, off by 2 users out of 13,979,592 |
| MDE (pre-specified) | 5% relative |
| Required N | 1,949,765 control users at the 85/15 allocation |
| Variance removed by CUPED | 10.70% |
| CI width removed by CUPED | 5.50% |
| Effect among users actually exposed | +3.196 pp |

---

## What the pipeline does

```
scripts/get_data.sh   fetch and decompress the raw log
src/ingest.py         [1]  Spark: 13.9M rows -> per-arm sufficient statistics
src/srm.py            [2]  sample ratio mismatch, the count guardrail
src/balance.py        [2b] covariate balance, the composition guardrail
src/power.py          [3]  MDE and required N, computed before the effect
src/analyze.py        [4]  two-proportion z-test, CUPED, Lin, CACE
src/run.py            [5]  orchestration, verdict, results/summary.json
tests/test_stats.py        22 tests against known-answer synthetic data
```

Guardrails run before the effect is computed and the run exits non-zero if SRM
fails. Power runs before that, because a design question you answer after
seeing the result is not a design question any more.

### Spark's actual job

Every test here touches the raw 13.9M rows only through a handful of sums.
The z-test needs `n` and `sum(Y)` per arm. CUPED needs
`theta = Cov(Y,X)/Var(X)`, which comes out of `n`, `sum(X)`, `sum(X²)`,
`sum(Y)` and `sum(XY)`. All additive, all computable in one pass.

So Spark computes the sums and the driver never receives more than a couple of
dozen numbers plus a 12×12 cross-product matrix. That is what makes
"aggregated 13.9M rows" a real statement rather than a synonym for "read a CSV".

---

## The three things worth reading

### 1. A perfect SRM pass is a warning sign

The split is 85.00/15.00, off by **2 users out of 13,979,592**. Chi-square
p = 0.9989. Green tick.

Except a chi-square p-value is uniform under the null, so **p = 0.9989 is
exactly as improbable as p = 0.0011**. One standard deviation of Bernoulli
noise at this n is 1,335 users, and the arms sit 0.0013 sd apart. Independent
random assignment lands that close well under 1% of the time.

The split is almost certainly *constructed* — control downsampled to exactly
15% after collection. Supporting evidence: exact-duplicate rows run 10.19% in
treatment against 2.34% in control. That 4x gap looks like a logging fault and
isn't one. Downsampling an arm to fraction *q* keeps a duplicate *pair* with
probability *q²*, so duplicate share scales with *q*: 10.19% × (0.15/0.85) =
1.8%, against 2.34% observed.

So the pipeline reports under-dispersion alongside the pass. A bare green tick
would imply this validates Criteo's randomiser, when all it demonstrates is
that the guardrail runs.

The stage also prints what the same counts look like tested against other
assumed designs, because that choice is the whole test:

| assumed share | chi² | p | |
|---|---|---|---|
| 50% | 6,850,005 | 0 | FAIL |
| 84% | 10,402 | 0 | FAIL |
| **85%** | **0.0** | **0.999** | **pass** |
| 86% | 11,611 | 0 | FAIL |

Test an 85/15 experiment against a 50/50 default and you get a catastrophic
p-value on a perfectly healthy experiment. The expected ratio has to come from
the design document.

### 2. SRM passes on counts. The arms still aren't comparable.

SRM asks whether the arms have the right *number* of users. It says nothing
about whether they contain the same *kind*. Here the counts are perfect and the
composition is not:

```
Hotelling T² = 5,142.5  ->  F(12, 13,979,579) = 428.55,  p = 0
all 12 features imbalanced, largest |SMD| = 0.0488 on f3
```

Every one of the twelve pre-assignment features differs between arms at
overwhelming significance — f3 at z = −67 — while **every standardised mean
difference sits under 0.05**, half the conventional 0.1 "negligible" threshold.
Statistically undeniable, practically trivial by normal standards.

It is not trivial here, because the effect is small. The CUPED adjustment term
is θ × imbalance = 1.2466 × 0.000135 = 0.000168, which is **14.6% of a 0.00115
effect**. A negligible imbalance eats a sixth of a small effect.

That is the finding: this experiment would pass every guardrail most teams run,
and its arms are still not exchangeable.

### 3. "CUPED must not move the point estimate" needs a yardstick

The rule as usually stated has no scale attached, and that gap cost a test
failure during the build. An assertion that CUPED moves the estimate by less
than 5% failed at 16% on synthetic data that was behaving perfectly.

Nothing was wrong. Under perfect randomisation the arms still differ on the
covariate by sampling noise, CUPED corrects that difference, and the size of
the correction is θ × se(imbalance) — independent of how big the effect is. A
small effect therefore moves a long way in percentage terms for entirely
innocent reasons.

The right benchmark is the shift against that chance figure:

```
point estimate moved     -14.610%   (chance alone would move it 1.191%)
shift vs chance          12.3x      (~1x is a healthy experiment)
```

12.3x is not noise. It is the same 12.3 as the z-score of the covariate
imbalance, as it has to be. That is why the adjusted estimate leads: the
covariate is fixed before assignment, so the adjustment corrects the imbalance
instead of inheriting it.

---

## Choices, and why

**Relative MDE, not absolute.** Control converts at 0.1938%. A 1-percentage-point
absolute MDE — the figure in the project brief — means detecting a 516% lift.
The sensitivity table makes the point concretely: a 1% *relative* MDE would need
318 million users, 23x the log.

| relative MDE | control users needed | total needed | achieved power | |
|---|---|---|---|---|
| 1% | 47,801,668 | 318,678,061 | 9.0% | underpowered |
| 2% | 12,009,536 | 80,063,643 | 21.6% | underpowered |
| **5%** | **1,949,765** | **12,998,445** | **82.8%** | ok |
| 10% | 499,098 | 3,327,323 | 100% | ok |
| 20% | 130,508 | 870,055 | 100% | ok |

5% was pre-specified in `config.py` before any effect was computed. Inverting
the power curve, the smallest lift these arms can resolve at 80% power is
**4.82% relative** — so the pre-specified target sits just inside the floor.

**An 85/15 split costs nearly double the users.** Required N at an unequal
allocation is `n_balanced × (2 + k + 1/k) / 2` against `2 × n_balanced`
balanced. At k = 5.67 that is **1.96x**. `Var(diff)` goes as `1/n_t + 1/n_c` and
the smaller arm dominates the sum, so skewing the split to save on control
users costs precision fast. Worth knowing before agreeing to one.

**`visit` is the wrong CUPED covariate, and the pipeline shows why.** The brief
suggests it. It is measured *during* the experiment and the treatment moves it
by a full percentage point (p = 0), so adjusting on it removes part of the
causal path. Run anyway, as a labelled contrast:

```
CUPED on visit:  variance reduction 4.96%,  point estimate moved -55.73%
```

A tighter interval around a differently-biased number. The covariate used
instead is an OLS index over `f0..f11` — user attributes fixed before
assignment — fitted on a held-out control-only sample so the treatment effect
stays out of the coefficients. R² = 0.107.

**Variance reduction and CI width reduction are different numbers.** CI width
scales with the square root of variance, so 10.70% variance removed is
`1 − √(1 − 0.107)` = **5.50%** off the interval. Quoting the variance figure as
an interval improvement roughly doubles the claim. Both are printed, labelled.

**Intent-to-treat, not treated-on-treated.** Only 3.6% of the treatment arm was
ever exposed to an ad, and control exposure is exactly 0. Conditioning on
exposure would condition on a post-randomisation variable and reintroduce the
selection bias randomisation removed. ITT is the headline. Because the
noncompliance is one-sided, the Wald estimator is valid and gives the effect
among users actually reached: **+3.196 pp** [+3.009, +3.383], about 28x the ITT.

**Lin's estimator as a robustness check.** Plain CUPED fits one θ to both arms,
assuming the covariate relates to the outcome identically under treatment and
control. Per-arm θ comes out 1.289 and 0.997 — a real treatment-by-covariate
interaction — so a single pooled θ was misfitting both arms. Lin gives +52.20%
against CUPED's +50.76%, close enough to each other and far enough from the
unadjusted +59.45% to make the picture clear.

---

## What this does not establish

The arms differ on every observed pre-assignment feature. Adjustment fixes the
observed part. Nothing fixes the unobserved part.

So the true effect is *bracketed* by these estimates rather than pinned by
either. Somewhere between +50% and +59% relative, with the adjusted end more
credible and residual confounding from unmeasured features not ruled out. A
clean experiment would not need that sentence, and saying so is more useful
than a point estimate with false precision.

Two further limits worth naming. The adjustment uses a single linear index over
the twelve features; a full ANCOVA on all twelve separately would be the next
step, and it needs twelve more per-arm sums from Spark. And there is no user ID
in the file, so "one row per user" cannot be verified by key — the 1,259,545
exact duplicate rows are reported rather than dropped, since removing them
would silently move the very split ratio SRM exists to test.

---

## Setup

Needs Python 3.12 and **JDK 17**. PySpark 3.5 does not run on JDK 21+ — module
encapsulation blocks its reflective access to `DirectBuffer` and it dies before
any of your code executes. `src/ingest.py` resolves JDK 17 through
`/usr/libexec/java_home` rather than trusting `JAVA_HOME`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/get_data.sh
.venv/bin/python -m src.run
```

Dataset: Criteo Uplift Prediction v2.1, CC BY-NC-SA 4.0, from
Diemert, Betlei, Renaudin & Amini, *A Large Scale Benchmark for Uplift
Modeling* (AdKDD 2018). The URL in the paper is dead; the file is mirrored at
`huggingface.co/datasets/criteo/criteo-uplift` and the fetch script verifies
its SHA256.
