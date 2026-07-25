"""One command, the whole pipeline, and a verdict at the end.

    python -m src.run              full run, Spark included
    python -m src.run --skip-spark reuse the existing aggregate (stats only)

Stage order is not arbitrary. Guardrails run before the effect is computed, and
the power analysis runs before that, because a design question you answer after
seeing the result is not a design question any more. If the guardrails fail
hard the run stops rather than printing a number nobody should act on.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from src import analyze, balance, config, power, srm


BAR = "=" * 78


def header(n: str, title: str) -> None:
    print(f"\n{BAR}\n{n}  {title}\n{BAR}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GroundTruth experiment engine")
    ap.add_argument("--skip-spark", action="store_true",
                    help="reuse the existing Spark aggregate instead of re-reading the raw log")
    ap.add_argument("--mde", type=float, default=config.MDE_RELATIVE,
                    help="minimum detectable effect, as a relative lift (default 0.05)")
    ap.add_argument("--expected-share", type=float, default=config.DESIGN_TREATMENT_SHARE,
                    help="designed treatment share for the SRM test (default 0.85)")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(BAR)
    print("GroundTruth - A/B test analysis on the Criteo Uplift log")
    print(BAR)

    # --- Stage 1 -----------------------------------------------------------
    header("[1/5]", "INGEST - Spark, raw log to sufficient statistics")
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

    # --- Stage 2 -----------------------------------------------------------
    header("[2/5]", "SRM - do the arms have the right number of users?")
    srm_res = srm.check(n_t, n_c, expected_treatment_share=args.expected_share)
    srm.report(srm_res, srm.sensitivity(n_t, n_c))

    if not srm_res.passed:
        print("\n  !! SRM FAILED. Downstream effect estimates are not trustworthy.")
        print("  !! Fix assignment or logging before reading any number below.")
        print(f"\n  {srm_res.verdict}")
        return 1

    # --- Stage 2b ----------------------------------------------------------
    header("[2b/5]", "BALANCE - do the arms contain the same kind of user?")
    bal_res = balance.check(cells)
    balance.report(bal_res)

    # --- Stage 3 -----------------------------------------------------------
    header("[3/5]", "POWER - computed before the effect is looked at")
    pow_res = power.analyse(
        baseline_rate=arms["control"]["conversion_rate"],
        n_treatment=n_t, n_control=n_c, mde_relative=args.mde,
    )
    power.report(pow_res)

    # --- Stage 4 -----------------------------------------------------------
    header("[4/5]", "EFFECT - unadjusted, then CUPED")
    ana = analyze.analyse(cells)
    analyze.report(ana)

    # --- Stage 5 -----------------------------------------------------------
    header("[5/5]", "VERDICT")
    summary = build_summary(ingest_out, srm_res, bal_res, pow_res, ana, args, t0)
    print_verdict(summary)

    config.SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {config.SUMMARY_JSON.relative_to(config.REPO_ROOT)}  "
          f"({time.time() - t0:.1f}s total)")
    return 0


def build_summary(ingest_out, srm_res, bal_res, pow_res, ana, args, t0) -> dict:
    """Every number the project claims, in one file, all of it reproducible."""
    u = ana["unadjusted"]
    c = ana["cuped"]

    # Which estimate leads. If the arms are balanced, the unadjusted difference
    # is already unbiased and CUPED is a pure precision gain. If they are not,
    # the adjusted estimate is the one that corrects for the imbalance instead
    # of inheriting it, and the unadjusted number should not be the headline.
    balanced = bal_res.hotelling_p >= 0.001
    recommended = "unadjusted" if balanced else "cuped"

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Criteo Uplift Prediction v2.1",
            "source": "huggingface.co/datasets/criteo/criteo-uplift",
            "rows": ingest_out["total_rows"],
            "columns": ingest_out["schema"],
            "exact_duplicate_rows": ingest_out["quality"]["exact_duplicate_rows"],
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
        },
        "recommended_estimate": recommended,
        "recommendation_reason": (
            "Arms are balanced on observed pre-assignment features, so the "
            "unadjusted difference is already unbiased and CUPED is a pure "
            "precision gain."
            if balanced else
            "Arms are imbalanced on pre-assignment features (Hotelling "
            f"p={bal_res.hotelling_p:.2e}), so the unadjusted difference "
            "inherits that imbalance. The covariate-adjusted estimate corrects "
            "for it and is the defensible headline."
        ),
        "ingest": ingest_out,
        "srm": srm_res.as_dict(),
        "balance": bal_res.as_dict(),
        "power": pow_res.as_dict(),
        "effect": ana,
        "config": {
            "alpha": config.ALPHA,
            "power_target": config.POWER,
            "mde_relative": args.mde,
            "design_treatment_share": args.expected_share,
            "metric": config.METRIC,
            "covariate": config.COVARIATE_SAFE,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }


def print_verdict(s: dict) -> None:
    h = s["headline"]
    u = s["effect"]["unadjusted"]
    c = s["effect"]["cuped"]["test"]
    lead = c if s["recommended_estimate"] == "cuped" else u

    print(f"  SRM_P       {h['SRM_P']:.6f}          chi-square p on the split")
    print(f"  RATIO       {h['RATIO']:<18} observed treatment/control split")
    print(f"  MDE         {h['MDE']:.0%}{'':<16} relative lift, pre-specified")
    print(f"  N           {h['N']:,}{'':<10} control users required at 85/15")
    print(f"  LIFT        {h['LIFT']:+.2%}{'':<12} unadjusted relative lift")
    print(f"  LIFT_CUPED  {h['LIFT_CUPED']:+.2%}{'':<12} CUPED-adjusted relative lift")
    print(f"  CUPED_PCT   {h['CUPED_PCT']:.2%}{'':<13} variance removed by CUPED")
    print(f"              {h['CUPED_CI_WIDTH_PCT']:.2%}{'':<13} CI width removed by CUPED")
    print(f"  FINAL_P     {h['FINAL_P']:.3e}{'':<10} p after CUPED adjustment")

    ship = lead["p_value"] < config.ALPHA and lead["ci_low"] > 0
    print()
    print(f"  ship/no-ship  {'SHIP' if ship else 'DO NOT SHIP'}")
    print(f"  leading with  {s['recommended_estimate']} "
          f"({lead['relative_lift']:+.2%}, 95% CI "
          f"[{lead['relative_ci_low']:+.2%}, {lead['relative_ci_high']:+.2%}])")

    import textwrap
    print()
    print(textwrap.fill(f"why: {s['recommendation_reason']}", 78,
                        initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill(
        "caveat: the arms differ on every observed pre-assignment feature. "
        "Adjustment fixes the observed part. Nothing fixes the unobserved "
        "part, so the true effect is bracketed by these estimates rather than "
        "pinned by either of them.", 78,
        initial_indent="  ", subsequent_indent="  "))


if __name__ == "__main__":
    sys.exit(main())
