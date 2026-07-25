"""Shared paths and experiment design constants.

Everything the pipeline treats as a *decision* lives here, so the decisions are
reviewable in one place instead of buried in four scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = REPO_ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv"
RAW_GZ = REPO_ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
AGG_DIR = REPO_ROOT / "data" / "aggregate"
CELL_PARQUET = AGG_DIR / "cells.parquet"
INGEST_JSON = REPO_ROOT / "results" / "ingest.json"
SUMMARY_JSON = REPO_ROOT / "results" / "summary.json"

FEATURES = [f"f{i}" for i in range(12)]
RAW_COLUMNS = FEATURES + ["treatment", "conversion", "visit", "exposure"]

# --- Experiment design -----------------------------------------------------

# Criteo's own paper (Diemert et al., AdKDD 2018) states the log was collected
# under a randomised split with ~85% of users in treatment. That is the design
# ratio the SRM check tests against. Testing against 50/50 here would be wrong:
# it would "fail" an experiment that was never meant to be balanced, which is
# the single most common way an SRM check gets misused.
DESIGN_TREATMENT_SHARE = 0.85

ALPHA = 0.05
POWER = 0.80

# Minimum detectable effect, expressed as a RELATIVE lift on the baseline
# conversion rate. Criteo's baseline conversion is a fraction of a percent, so
# an absolute MDE (e.g. "1 percentage point") would be asking to detect a
# several-hundred-percent lift. Relative is the only sane framing at this
# baseline.
MDE_RELATIVE = 0.05
MDE_SENSITIVITY = [0.01, 0.02, 0.05, 0.10, 0.20]

# Primary metric and the CUPED covariate.
#
# `visit` is not a true pre-experiment covariate - it is measured during the
# experiment and is itself affected by treatment. Using it violates CUPED's
# core assumption and biases the estimate. `f0`..`f11` are user features fixed
# before assignment, so they are legitimate covariates. We compute both and
# report the difference, because the contrast is the point.
METRIC = "conversion"
COVARIATE_SAFE = "f_index"   # pre-assignment feature composite, built in ingest
COVARIATE_NAIVE = "visit"    # the tempting-but-wrong choice, kept for contrast

# Folds for cross-fitting the covariate model. Every row's covariate comes from
# a model fitted without it, so the covariate cannot be overfit to the rows it
# later adjusts.
COVARIATE_FOLDS = 2

# --- Decision rule ---------------------------------------------------------

# The minimum lift that would make this campaign worth shipping, as a relative
# effect on the baseline. A p-value answers "is the effect distinguishable from
# zero"; it does not answer "is the effect worth having". Those are different
# questions and only the second one is a decision.
#
# Criteo publishes no cost data, so this is a stated assumption rather than a
# measured break-even, and the pipeline says so wherever it uses it. The point
# is that the number exists and is visible, not that it is authoritative.
MIN_PRACTICAL_LIFT = 0.10

SPARK_APP_NAME = "GroundTruth-ExperimentEngine"
SPARK_MASTER = os.environ.get("GT_SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.environ.get("GT_SPARK_DRIVER_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = os.environ.get("GT_SPARK_SHUFFLE_PARTITIONS", "16")
