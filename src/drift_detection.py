"""
drift_detection.py
------------------
Core drift-detection engine: statistical tests, quality checks, and scoring.

Functions:
    run_quality_checks    – Detect missing-value spikes and range violations
    run_drift_tests       – Kolmogorov-Smirnov two-sample test per feature
    compute_reliability_score – Composite 0-100 data-health score
    classify_score        – Map score to human-readable risk label
    run_full_analysis     – Orchestrate all checks and return a single report
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Thresholds (configurable)
# ---------------------------------------------------------------------------
MISSING_SPIKE_THRESHOLD = 0.05   # flag if today's missing% exceeds baseline by >5%
KS_P_VALUE_THRESHOLD = 0.05     # reject H0 (same distribution) at 5% significance
PENALTY_MISSING = 5             # points deducted per missing-spike flag
PENALTY_RANGE = 5               # points deducted per range-violation flag
PENALTY_DRIFT = 10              # points deducted per confirmed drift


def run_quality_checks(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    baseline_profile: Dict[str, Dict[str, float]],
    columns: list,
) -> pd.DataFrame:
    """
    Check each feature for missing-value spikes and range boundary violations.

    A **missing spike** is flagged when the current dataset's missing percentage
    exceeds the baseline's by more than MISSING_SPIKE_THRESHOLD (default 5%).

    A **range violation** is flagged when the current dataset contains values
    outside the baseline's [min, max] envelope.

    Parameters
    ----------
    baseline : pd.DataFrame
    current : pd.DataFrame
    baseline_profile : dict
        Output of data_loader.build_baseline_profile().
    columns : list[str]
        Numeric columns to check.

    Returns
    -------
    pd.DataFrame
        One row per feature with boolean columns: missing_spike, range_issue.
    """
    rows = []
    for col in columns:
        current_missing = float(current[col].isna().mean())
        baseline_missing = baseline_profile[col]["missing_pct"]
        missing_spike = (current_missing - baseline_missing) > MISSING_SPIKE_THRESHOLD

        col_min = current[col].min()
        col_max = current[col].max()
        range_issue = bool(
            col_min < baseline_profile[col]["min"]
            or col_max > baseline_profile[col]["max"]
        )

        rows.append(
            {
                "feature": col,
                "baseline_missing_pct": round(baseline_missing * 100, 2),
                "current_missing_pct": round(current_missing * 100, 2),
                "missing_spike": missing_spike,
                "baseline_min": baseline_profile[col]["min"],
                "baseline_max": baseline_profile[col]["max"],
                "current_min": float(col_min) if not pd.isna(col_min) else None,
                "current_max": float(col_max) if not pd.isna(col_max) else None,
                "range_issue": range_issue,
            }
        )
    return pd.DataFrame(rows)


def run_drift_tests(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    columns: list,
) -> pd.DataFrame:
    """
    Run a Kolmogorov-Smirnov two-sample test on each feature to detect
    distribution drift between the baseline and current datasets.

    The KS test is non-parametric and compares the empirical CDFs of two
    samples. A low p-value (< KS_P_VALUE_THRESHOLD) indicates that the two
    distributions are statistically different.

    Parameters
    ----------
    baseline : pd.DataFrame
    current : pd.DataFrame
    columns : list[str]

    Returns
    -------
    pd.DataFrame
        One row per feature with: ks_statistic, p_value, drift_detected.
    """
    rows = []
    for col in columns:
        baseline_vals = baseline[col].dropna()
        current_vals = current[col].dropna()

        if len(baseline_vals) < 2 or len(current_vals) < 2:
            # Not enough data to run the test
            rows.append(
                {
                    "feature": col,
                    "ks_statistic": None,
                    "p_value": None,
                    "drift_detected": False,
                }
            )
            continue

        stat, p_value = ks_2samp(baseline_vals, current_vals)
        rows.append(
            {
                "feature": col,
                "ks_statistic": round(float(stat), 6),
                "p_value": float(p_value),
                "drift_detected": p_value < KS_P_VALUE_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def compute_reliability_score(quality_df: pd.DataFrame, drift_df: pd.DataFrame) -> int:
    """
    Compute a composite Data Reliability Score (0–100).

    Scoring formula:
        Start at 100
        − 5  for each feature with a missing-value spike
        − 5  for each feature with a range violation
        − 10 for each feature where drift is confirmed (KS p < 0.05)

    Parameters
    ----------
    quality_df : pd.DataFrame
        Output of run_quality_checks().
    drift_df : pd.DataFrame
        Output of run_drift_tests().

    Returns
    -------
    int
        Clamped to [0, 100].
    """
    score = 100
    score -= int(quality_df["missing_spike"].sum()) * PENALTY_MISSING
    score -= int(quality_df["range_issue"].sum()) * PENALTY_RANGE
    score -= int(drift_df["drift_detected"].sum()) * PENALTY_DRIFT
    return max(0, min(100, score))


def classify_score(score: int) -> Tuple[str, str]:
    """
    Map a reliability score to a risk label and emoji indicator.

    Returns
    -------
    (label, color)
        label : str   – "Safe for ML", "Review Recommended", or "Unreliable"
        color : str   – CSS-friendly colour name for Streamlit badges
    """
    if score >= 85:
        return "✅ Safe for ML / Analytics", "green"
    elif score >= 60:
        return "⚠️ Review Recommended", "orange"
    else:
        return "❌ Unreliable — Do Not Use for ML", "red"


def run_full_analysis(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    baseline_profile: Dict[str, Dict[str, float]],
    columns: list,
) -> Tuple[pd.DataFrame, int, str, str]:
    """
    Orchestrate the complete drift-detection pipeline.

    Returns
    -------
    report : pd.DataFrame
        Merged quality + drift report with a per-feature status column.
    score : int
        Data Reliability Score (0–100).
    label : str
        Human-readable risk classification.
    color : str
        Associated colour for dashboard rendering.
    """
    quality_df = run_quality_checks(baseline, current, baseline_profile, columns)
    drift_df = run_drift_tests(baseline, current, columns)

    # Merge into a single report table
    report = quality_df.merge(drift_df, on="feature")
    report["status"] = report.apply(
        lambda r: "⚠ Issue"
        if (r["missing_spike"] or r["range_issue"] or r["drift_detected"])
        else "✅ Stable",
        axis=1,
    )

    score = compute_reliability_score(quality_df, drift_df)
    label, color = classify_score(score)

    return report, score, label, color
