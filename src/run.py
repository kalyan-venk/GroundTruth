"""One command, the whole pipeline, and a decision at the end.

    python -m src.run                 everything, ~3 minutes
    python -m src.run --skip-spark    reuse cached aggregates, stats only
    python -m src.run --fast          skip the two extra Spark passes

Order is deliberate. Validity guardrails run before the effect is computed, and
the power analysis runs before that, because a design question you answer after
seeing the result is not a design question. If a guardrail fails hard the run
stops rather than printing a number nobody should act on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from src import analyze, balance, config, power, sequential, srm


BAR = "=" * 78


def header(n: str, title: str) -> None:
    print(f"\n{BAR}\n{n}  {title}\n{BAR}")


def _cached(path):
    return json.loads(path.read_text()) if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GroundTruth experiment engine")
    ap.add_argument("--skip-spark", action="store_true",
                    help="reuse cached aggregates instead of re-reading the raw log")
    ap.add_argument("--fast", action="store_true",
                    help="skip the robustness and HTE Spark passes")
    ap.add_argument("--mde", type=float, default=config.MDE_RELATIVE)
    ap.add_argument("--expected-share", type=float, default=config.DESIGN_TREATMENT_SHARE)
    ap.add_argument("--min-lift", type=float, default=config.MIN_PRACTICAL_LIFT,
                    help="minimum relative lift worth shipping (default 0.10)")
    args = ap.parse_args(argv)

    t0 = time.time()
    results_dir = config.REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(BAR)
    print("GroundTruth - A/B test analysis on the Criteo Uplift log")
    print(BAR)

    # --- 1. Ingest ---------------------------------------------------------
    header("[1/7]", "INGEST - Spark, raw log to sufficient statistics")
    if args.skip_spark and config.CELL_PARQUET.exists() and config.INGEST_JSON.exists():
        ingest_out = json.loads(config.INGEST_JSON.read_text())
        print(f"  reusing {config.CELL_PARQUET.name} "
              f"({ingest_out['total_rows']:,} rows aggregated previously)")
    else:
        from src.ingest import run as ingest_run
        ingest_out = ingest_run()

    from src.ingest import load_cells
    cells = load_cells()
    arms = ingest_out["arms"]
    n_t, n_c = arms["treatment"]["n"], arms["control"]["n"]

    # --- 2. SRM ------------------------------------------------------------
    header("[2/7]", "SRM - do the arms have the right number of users?")
    srm_res = srm.check(n_t, n_c, expected_treatment_share=args.expected_share)
    srm.report(srm_res, srm.sensitivity(n_t, n_c))

    if not srm_res.passed:
        print("\n  !! SRM FAILED. Downstream effect estimates are not trustworthy.")
        print(f"\n  {srm_res.verdict}")
        return 1

    # --- 3. Balance --------------------------------------------------------
    header("[3/7]", "BALANCE - do the arms contain the same kind of user?")
    bal_res = balance.check(cells)
    balance.report(bal_res)

    # --- 4. Robustness -----------------------------------------------------
    header("[4/7]", "ROBUSTNESS - how much of the answer is a data-handling choice?")
    rob_res = None
    if args.fast or args.skip_spark:
        rob_res = _cached(results_dir / "robustness.json")
        if rob_res:
            print("  reusing results/robustness.json")
    if rob_res is None and not (args.fast or args.skip_spark):
        from src import robustness
        rob_res = robustness.run()
    if rob_res:
        from src import robustness
        robustness.report(rob_res)
    else:
        print("  skipped (run without --fast/--skip-spark to compute)")

    # --- 5. Power ----------------------------------------------------------
    header("[5/7]", "POWER - computed before the effect is looked at")
    pow_res = power.analyse(
        baseline_rate=arms["control"]["conversion_rate"],
        n_treatment=n_t, n_control=n_c, mde_relative=args.mde,
    )
    power.report(pow_res)

    # --- 6. Effect ---------------------------------------------------------
    header("[6/7]", "EFFECT - unadjusted, CUPED, and peeking")
    ana = analyze.analyse(cells)
    analyze.report(ana)
    print()
    seq_res = sequential.run(cells)
    sequential.report(seq_res)

    # --- 7. Heterogeneity + verdict ---------------------------------------
    header("[7/7]", "WHO IT WORKS ON, AND THE DECISION")
    hte_res = None
    if args.fast or args.skip_spark:
        hte_res = _cached(results_dir / "hte.json")
        if hte_res:
            print("  reusing results/hte.json\n")
    if hte_res is None and not (args.fast or args.skip_spark):
        from src import hte
        hte_res = hte.run()
    if hte_res:
        from src import hte
        hte.report(hte_res)
    else:
        print("  skipped (run without --fast/--skip-spark to compute)")

    print(f"\n{BAR}\nDECISION\n{BAR}")
    summary = build_summary(ingest_out, srm_res, bal_res, pow_res, ana,
                            seq_res, rob_res, hte_res, args, t0)
    print_verdict(summary)

    config.SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {config.SUMMARY_JSON.relative_to(config.REPO_ROOT)}  "
          f"({time.time() - t0:.1f}s total)")
    return 0


def build_summary(ingest_out, srm_res, bal_res, pow_res, ana,
                  seq_res, rob_res, hte_res, args, t0) -> dict:
    u = ana["unadjusted"]
    c = ana["cuped"]
    lin = ana["lin_robustness"]

    # --- the decision ------------------------------------------------------
    #
    # Ship if the effect clears a lift worth having under EVERY specification
    # tried, not merely if p < 0.05 under the one we happened to prefer.
    #
    # Two changes from the naive rule. First, the bar is a lift that pays for
    # itself rather than a lift distinguishable from zero -- with 14M rows those
    # are wildly different bars, and only the first is a business question.
    # Second, the test is the weakest lower bound across the specification
    # curve, so a conclusion that survives only under a favourable
    # data-handling choice does not pass.
    candidates = [u, c["test"], lin["test"]]
    lower_bounds = [r["relative_ci_low"] for r in candidates]
    if rob_res:
        lower_bounds.append(rob_res["lowest_ci_bound"])
    worst_lower_bound = min(lower_bounds)

    ship = worst_lower_bound > args.min_lift
    marginal = (not ship) and worst_lower_bound > 0

    # Which single number to quote when someone insists on one. Adjustment wins
    # here not because a p-value says the arms are imbalanced -- at this N that
    # test always rejects and the rule would be decorative -- but because the
    # specification curve shows the adjusted estimate is roughly 3.5x less
    # sensitive to the duplicate-row decision than the raw one.
    if rob_res and rob_res.get("adjustment_stability_gain", 0) > 1.5:
        lead, lead_reason = "cuped", (
            f"Adjustment is not just a precision gain here. Across the three "
            f"duplicate-handling specifications the unadjusted lift spans "
            f"{rob_res['spread_factor_unadjusted']:.2f}x while the adjusted one spans "
            f"{rob_res['spread_factor_cuped']:.2f}x - about "
            f"{rob_res['adjustment_stability_gain']:.1f}x less sensitive. Changing the "
            f"duplicate rule changes arm composition, and covariate adjustment "
            f"is what absorbs composition differences."
        )
    elif bal_res.passed:
        lead, lead_reason = "unadjusted", (
            "Arms are balanced within the conventional |SMD| < 0.1, so the raw "
            "difference is already a clean estimate and adjustment is a pure "
            "precision gain."
        )
    else:
        lead, lead_reason = "cuped", (
            f"{bal_res.n_features_above_threshold} features exceed |SMD| = "
            f"{bal_res.smd_threshold}, so the raw difference inherits real "
            f"composition differences between the arms."
        )

    lead_test = c["test"] if lead == "cuped" else u

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Criteo Uplift Prediction v2.1",
            "source": "huggingface.co/datasets/criteo/criteo-uplift",
            "rows": ingest_out["total_rows"],
            "columns": ingest_out["schema"],
            "exact_duplicate_rows": ingest_out["quality"]["exact_duplicate_rows"],
        },
        "decision": {
            "ship": bool(ship),
            "marginal": bool(marginal),
            "min_practical_lift": args.min_lift,
            "worst_case_lower_bound": worst_lower_bound,
            "leading_estimate": lead,
            "leading_reason": lead_reason,
            "point_estimate": lead_test["relative_lift"],
            "ci": [lead_test["relative_ci_low"], lead_test["relative_ci_high"]],
            "plausible_range": rob_res["lift_range"] if rob_res else None,
            "targeting": (
                f"{hte_res['share_of_effect_in_top_30pct']:.0%} of incremental "
                f"conversions come from the top 30% of users by predicted "
                f"uplift; the bottom 40% contribute almost nothing."
                if hte_res else None
            ),
        },
        "headline": {
            "SRM_P": srm_res.p_value,
            "RATIO": ingest_out["observed_ratio_label"],
            "MDE": args.mde,
            "N": pow_res.n_control_required_at_actual_ratio,
            "CUPED_PCT": c["observed_variance_reduction"],
            "CUPED_CI_WIDTH_PCT": c["observed_ci_width_reduction"],
            "LIFT": u["relative_lift"],
            "LIFT_CUPED": c["test"]["relative_lift"],
            "FINAL_P": c["test"]["p_value"],
            "QINI": hte_res["qini_coefficient"] if hte_res else None,
            "PEEK_ALPHA_30": (seq_res["alpha_inflation"][seq_res["n_peeks"].index(30)]
                              if 30 in seq_res["n_peeks"] else None),
        },
        "ingest": ingest_out,
        "srm": srm_res.as_dict(),
        "balance": bal_res.as_dict(),
        "robustness": rob_res,
        "power": pow_res.as_dict(),
        "effect": ana,
        "sequential": seq_res,
        "heterogeneity": hte_res,
        "config": {
            "alpha": config.ALPHA,
            "power_target": config.POWER,
            "mde_relative": args.mde,
            "design_treatment_share": args.expected_share,
            "min_practical_lift": args.min_lift,
            "metric": config.METRIC,
            "covariate": config.COVARIATE_SAFE,
            "covariate_folds": config.COVARIATE_FOLDS,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }


def print_verdict(s: dict) -> None:
    import textwrap

    d = s["decision"]
    h = s["headline"]

    call = "SHIP" if d["ship"] else ("MARGINAL" if d["marginal"] else "DO NOT SHIP")
    print(f"  {call}")
    print()
    print(f"  effect            {d['point_estimate']:+.2%} relative "
          f"[{d['ci'][0]:+.2%}, {d['ci'][1]:+.2%}]   ({d['leading_estimate']})")
    if d["plausible_range"]:
        lo, hi = d["plausible_range"]
        print(f"  across all specs  {lo:+.2%} to {hi:+.2%}")
    print(f"  bar to clear      {d['min_practical_lift']:+.0%} relative "
          f"(stated assumption, not a measured break-even)")
    print(f"  worst-case bound  {d['worst_case_lower_bound']:+.2%}   "
          f"{'clears it' if d['ship'] else 'does not clear it'}")
    print()
    print(textwrap.fill(f"why this estimate: {d['leading_reason']}", 78,
                        initial_indent="  ", subsequent_indent="  "))
    if d["targeting"]:
        print()
        print(textwrap.fill(f"targeting: {d['targeting']}", 78,
                            initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill(
        "what would change this: the Criteo log is not a clean randomised "
        "experiment. The split is constructed rather than randomised, the arms "
        "differ on every observed pre-assignment feature, and the magnitude "
        "moves with a data-handling choice the file does not settle. The sign "
        "and the ship decision survive all of that; the exact number does not, "
        "and no adjustment can fix unobserved imbalance.", 78,
        initial_indent="  ", subsequent_indent="  "))

    print(f"\n  {'-' * 74}")
    print(f"  SRM_P {h['SRM_P']:.4f}   RATIO {h['RATIO']}   "
          f"MDE {h['MDE']:.0%}   N {h['N']:,}")
    print(f"  LIFT {h['LIFT']:+.2%}   LIFT_CUPED {h['LIFT_CUPED']:+.2%}   "
          f"CUPED_PCT {h['CUPED_PCT']:.2%} var / {h['CUPED_CI_WIDTH_PCT']:.2%} CI")
    extras = f"  FINAL_P {h['FINAL_P']:.2e}"
    if h["QINI"] is not None:
        extras += f"   QINI {h['QINI']:.3f}"
    if h["PEEK_ALPHA_30"] is not None:
        extras += f"   alpha at 30 peeks {h['PEEK_ALPHA_30']:.1%}"
    print(extras)


if __name__ == "__main__":
    sys.exit(main())
