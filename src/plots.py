"""Stage 8 - the five charts worth looking at.

Written to be read in ten seconds each, which is roughly how long anyone spends
on a chart in a README. Every one answers a single question named in its title,
carries its numbers as direct labels rather than making the reader measure
against an axis, and keeps the grid recessive so the marks are the thing you
see.

    python -m src.plots        writes results/figures/*.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config

FIG_DIR = config.REPO_ROOT / "results" / "figures"

# Categorical slots 1-3 of the reference palette, validated for colour-vision
# deficiency at all pairs (worst CVD dE 9.2, worst normal-vision dE 24.0).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"          # status only - thresholds and failure markers
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"


def _style(ax, title, subtitle=None):
    """Recessive frame. The data should be the only thing with weight."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=12, fontweight="600",
                 loc="left", pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2,
                fontsize=9.5, va="bottom")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# --- 1. Balance ------------------------------------------------------------

def love_plot(summary):
    """Standardised mean differences, sorted, against the 0.1 convention.

    A love plot is the standard way to show covariate balance because it puts
    every feature on one scale-free axis, so "is this experiment balanced" is a
    single glance rather than twelve numbers. The threshold lines are the whole
    point: without them a reader has no way to know whether 0.05 is fine.
    """
    rows = sorted(summary["balance"]["per_feature"], key=lambda d: d["smd"])
    names = [r["feature"] for r in rows]
    smds = [r["smd"] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _style(ax, "Every feature is imbalanced. None of them meaningfully.",
           "standardised mean difference, treatment minus control")

    ax.axvline(0, color=INK_3, linewidth=1, zorder=1)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=RED, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(0.1, len(rows) - 0.4, "  |SMD| = 0.1\n  negligible threshold",
            color=RED, fontsize=8.5, va="top")

    ax.hlines(y, 0, smds, color=GRID, linewidth=1.5, zorder=2)
    ax.scatter(smds, y, s=52, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    for yi, s in zip(y, smds):
        off = 0.006 if s > 0 else -0.006
        ax.text(s + off, yi, f"{s:+.4f}", color=INK_2, fontsize=8.5,
                va="center", ha="left" if s > 0 else "right")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.135, 0.135)
    ax.set_xlabel("standardised mean difference", color=INK_2, fontsize=9.5)
    return _save(fig, "01-balance.png")


# --- 2. Power --------------------------------------------------------------

def power_curve(summary):
    """Users required against the effect you want to detect.

    Log-scaled y because the relationship is roughly inverse-square: halving
    the detectable effect quadruples the sample. On a linear axis the useful
    part of the curve is invisible.
    """
    sens = summary["power"]["sensitivity"]
    mdes = [s["mde_relative"] for s in sens]
    need = [s["n_total_required"] for s in sens]
    have = summary["ingest"]["total_rows"]
    achieved = summary["power"]["mde_actually_detectable_relative"]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _style(ax, "Detecting a smaller effect costs superlinearly more users",
           f"alpha = 0.05, power = 0.80, baseline {summary['power']['baseline_rate']:.4%}")

    ax.plot(mdes, need, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.axhline(have, color=ORANGE, linewidth=2, linestyle="--", zorder=2)
    ax.text(0.205, have * 1.12, f"this experiment has {have/1e6:.1f}M users",
            color=ORANGE, fontsize=9, ha="right")

    ax.axvline(achieved, color=INK_3, linewidth=1, linestyle=":", zorder=2)
    ax.annotate(f"detection floor\n{achieved:.2%}",
                xy=(achieved, 0.80), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points",
                color=INK_2, fontsize=8.5, va="top")

    for m, n in zip(mdes, need):
        label = f"{n/1e6:.1f}M" if n > 1e6 else f"{n/1e3:.0f}k"
        ax.annotate(label, xy=(m, n), xytext=(0, 9), textcoords="offset points",
                    color=INK_2, fontsize=8.5, ha="center")

    ax.set_yscale("log")
    # Headroom above the tallest point so its label does not collide with the
    # subtitle. Log axis, so the padding is multiplicative.
    ax.set_ylim(min(need) / 2.2, max(need) * 3.0)
    ax.set_xlabel("minimum detectable effect, relative lift", color=INK_2, fontsize=9.5)
    ax.set_ylabel("users required, both arms", color=INK_2, fontsize=9.5)
    ax.set_xticks(mdes)
    ax.set_xticklabels([f"{m:.0%}" for m in mdes])
    return _save(fig, "02-power.png")


# --- 3. Forest -------------------------------------------------------------

def forest_plot(summary, ancova=None):
    e = summary["effect"]
    rows = [
        ("unadjusted", e["unadjusted"], BLUE),
        ("CUPED (1 index)", e["cuped"]["test"], ORANGE),
        ("Lin (per-arm slope)", e["lin_robustness"]["test"], ORANGE),
    ]
    if ancova:
        lo, hi = ancova["relative_ci_pooled"]
        rows.append(("ANCOVA, 12 covariates",
                     {"relative_lift": ancova["relative_lift_pooled"],
                      "relative_ci_low": lo, "relative_ci_high": hi},
                     ORANGE))
    rows.append(("CUPED on 'visit'\n(post-treatment - invalid)",
                 e["cuped_naive_contrast"]["test"], RED))

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    _style(ax, "Choice of adjustment moves the answer more than noise does",
           "relative lift in conversion, 95% intervals")

    y = np.arange(len(rows))[::-1]
    for yi, (label, r, color) in zip(y, rows):
        lo, hi = r["relative_ci_low"], r["relative_ci_high"]
        ax.hlines(yi, lo, hi, color=color, linewidth=2.5, zorder=3)
        ax.plot([lo, hi], [yi, yi], "|", color=color, markersize=9,
                markeredgewidth=2.5, zorder=3)
        ax.scatter([r["relative_lift"]], [yi], s=80, color=color, zorder=4,
                   edgecolor=SURFACE, linewidth=1.8)
        ax.text(hi + 0.012, yi, f"{r['relative_lift']:+.1%}", color=INK_2,
                fontsize=9, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("relative lift", color=INK_2, fontsize=9.5)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    return _save(fig, "03-forest.png")


# --- 4. Specification curve ------------------------------------------------

def specification_curve(robustness):
    """What the duplicate-row decision does to the answer.

    Two series on one axis on purpose. The finding is not either line's level,
    it is that one of them slopes steeply and the other barely moves.
    """
    specs = robustness["specifications"]
    labels = [f"{s['spec']}\n{s['n_total']/1e6:.1f}M rows" for s in specs]
    raw = [s["lift_unadjusted"] for s in specs]
    adj = [s["lift_cuped"] for s in specs]
    x = np.arange(len(specs))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _style(ax, "Adjustment is 3.5x less sensitive to the duplicate-row choice",
           "relative lift under three defensible readings of 1.26M duplicate rows")

    for series, color, name in ((raw, BLUE, "unadjusted"), (adj, ORANGE, "CUPED")):
        ax.plot(x, series, color=color, linewidth=2.5, marker="o", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3, label=name)
        for xi, v in zip(x, series):
            ax.text(xi, v + 0.022, f"{v:+.1%}", color=INK_2, fontsize=9,
                    ha="center")

    for xi, s in zip(x, specs):
        ok = s["srm_passed"]
        ax.text(xi, 0.34, "SRM pass" if ok else "SRM FAIL",
                color=INK_3 if ok else RED, fontsize=8.5, ha="center",
                fontweight="600" if not ok else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.30, 0.95)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("relative lift", color=INK_2, fontsize=9.5)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return _save(fig, "04-specification-curve.png")


# --- 5. Uplift -------------------------------------------------------------

def uplift_deciles(hte):
    """Where the incremental conversions actually come from.

    Bars for absolute uplift because that is what the budget buys, and a
    second panel for the cumulative share because "70% of spend could go" is
    the sentence the reader should leave with.
    """
    d = sorted(hte["deciles"], key=lambda x: -x["mean_score"])
    x = np.arange(len(d))
    uplift = [r["uplift"] * 100 for r in d]
    err = [r["se"] * 100 * 1.96 for r in d]

    total = sum(r["incremental_conversions"] for r in d)
    cum = np.cumsum([r["incremental_conversions"] for r in d]) / total

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.4),
                                   gridspec_kw={"height_ratios": [3, 2]})
    _style(ax1, "Almost all of the effect sits in the top tenth of users",
           "absolute uplift by decile of predicted uplift, best first")

    ax1.bar(x, uplift, width=0.62, color=BLUE, zorder=3)
    ax1.errorbar(x, uplift, yerr=err, fmt="none", ecolor=INK_3,
                 elinewidth=1.2, capsize=3, zorder=4)
    # Labels above the bars in ink, not inside them in white. Nine of these ten
    # bars are too short to hold text, and the tenth does not need the contrast
    # problem.
    for xi, v, e in zip(x, uplift, err):
        ax1.annotate(f"{v:+.3f}", xy=(xi, v + e), xytext=(0, 5),
                     textcoords="offset points", color=INK_2, fontsize=8.5,
                     ha="center")
    ax1.set_ylim(0, max(uplift) * 1.22)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"D{i+1}" for i in x], fontsize=9)
    ax1.set_ylabel("uplift, percentage points", color=INK_2, fontsize=9.5)

    _style(ax2, "", None)
    ax2.plot(np.arange(len(cum) + 1) / len(cum), np.concatenate([[0], cum]),
             color=ORANGE, linewidth=2.5, marker="o", markersize=6,
             markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    ax2.plot([0, 1], [0, 1], color=INK_3, linewidth=1, linestyle=":", zorder=2)
    ax2.text(0.70, 0.62, "random targeting", color=INK_3, fontsize=8.5,
             rotation=25, rotation_mode="anchor")
    ax2.axvline(0.3, color=INK_3, linewidth=1, linestyle="--", zorder=2)
    ax2.annotate(f"top 30% of users\ncarry {cum[2]:.0%} of the gain",
                 xy=(0.3, cum[2]), xytext=(-118, -58), textcoords="offset points",
                 color=INK_2, fontsize=9,
                 arrowprops=dict(arrowstyle="-", color=INK_3, linewidth=0.9))
    ax2.set_xlabel("share of users treated, best first", color=INK_2, fontsize=9.5)
    ax2.set_ylabel("share of incremental\nconversions", color=INK_2, fontsize=9.5)
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    return _save(fig, "05-uplift.png")


def run(verbose: bool = True) -> list:
    summary = json.loads(config.SUMMARY_JSON.read_text())
    results = config.REPO_ROOT / "results"

    def load(name):
        p = results / name
        return json.loads(p.read_text()) if p.exists() else None

    made = [love_plot(summary), power_curve(summary),
            forest_plot(summary, load("ancova.json"))]
    rob = summary.get("robustness") or load("robustness.json")
    if rob:
        made.append(specification_curve(rob))
    hte = summary.get("heterogeneity") or load("hte.json")
    if hte:
        made.append(uplift_deciles(hte))

    if verbose:
        for p in made:
            print(f"  wrote {p.relative_to(config.REPO_ROOT)}")
    return made


if __name__ == "__main__":
    run()
