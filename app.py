"""
app.py — Data Drift Monitoring System
======================================
Interactive Streamlit dashboard for detecting data quality issues and
statistical distribution drift between a baseline and incoming dataset.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Connect GitHub repo → set app.py as entrypoint.
"""

import streamlit as st
import pandas as pd
import pathlib

from src.data_loader import load_csv, validate_columns, build_baseline_profile
from src.drift_detection import run_full_analysis
from src.visualization import (
    plot_distribution_comparison,
    plot_reliability_gauge,
    plot_feature_status_bar,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Drift Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths to bundled sample data
DATA_DIR = pathlib.Path(__file__).parent / "data"
SAMPLE_BASELINE = DATA_DIR / "sample_baseline.csv"
SAMPLE_TODAY = DATA_DIR / "sample_today.csv"


# ---------------------------------------------------------------------------
# Sidebar — data source selection
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Data Drift Monitor")
st.sidebar.markdown("---")

data_mode = st.sidebar.radio(
    "Choose data source",
    ["Sample Data (pre-loaded)", "Upload Your Own CSV"],
    help="Try the sample data first to see how the tool works, then upload your own.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "1. Load a *baseline* dataset (training / reference data)\n"
    "2. Load a *current* dataset (new / production data)\n"
    "3. The tool runs quality checks and statistical drift tests\n"
    "4. A **Reliability Score** summarises overall data health"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by **[Mahi Sharma](https://github.com/mahi-sharmas)**  \n"
    "B.Tech CSE (Data Science), MUJ"
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_baseline = None
df_current = None

if data_mode == "Sample Data (pre-loaded)":
    if SAMPLE_BASELINE.exists() and SAMPLE_TODAY.exists():
        df_baseline = load_csv(str(SAMPLE_BASELINE))
        df_current = load_csv(str(SAMPLE_TODAY))
        st.info(
            "Using **sample credit-scoring data** (5 000 rows). "
            "The sample *current* dataset has injected drift — "
            "10% NaN spikes, 1.5× income scaling, and impossible age values — "
            "to demonstrate every detection capability.",
            icon="ℹ️",
        )
    else:
        st.error("Sample data files not found. Please check the `data/` folder.")
else:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        baseline_file = st.file_uploader(
            "Upload **Baseline** CSV",
            type=["csv"],
            help="Your reference / training dataset.",
        )
    with col_up2:
        current_file = st.file_uploader(
            "Upload **Current** CSV",
            type=["csv"],
            help="The new / production data to compare against the baseline.",
        )

    if baseline_file and current_file:
        df_baseline = load_csv(baseline_file)
        df_current = load_csv(current_file)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
st.title("📊 Data Drift & Quality Monitoring Dashboard")

if df_baseline is not None and df_current is not None:

    # Validate columns
    shared_cols, missing_cols = validate_columns(df_baseline, df_current)

    if missing_cols:
        st.warning(
            f"The current dataset is missing these baseline columns: "
            f"**{', '.join(missing_cols)}**. They will be skipped."
        )

    if not shared_cols:
        st.error("No shared numeric columns found between the two datasets.")
        st.stop()

    # Build baseline profile
    baseline_profile = build_baseline_profile(df_baseline)

    # ---- Data preview (collapsed by default) ----
    with st.expander("🔍 Preview the datasets", expanded=False):
        prev1, prev2 = st.columns(2)
        with prev1:
            st.markdown("**Baseline** — first 5 rows")
            st.dataframe(df_baseline.head(), use_container_width=True)
            st.caption(f"{df_baseline.shape[0]:,} rows × {df_baseline.shape[1]} columns")
        with prev2:
            st.markdown("**Current** — first 5 rows")
            st.dataframe(df_current.head(), use_container_width=True)
            st.caption(f"{df_current.shape[0]:,} rows × {df_current.shape[1]} columns")

    # ---- Run full analysis ----
    report, score, label, color = run_full_analysis(
        df_baseline, df_current, baseline_profile, shared_cols
    )

    # ---- Score + summary row ----
    st.markdown("---")
    score_col, label_col, counts_col = st.columns([1, 1.5, 1.5])

    with score_col:
        st.plotly_chart(
            plot_reliability_gauge(score, label),
            use_container_width=True,
        )

    with label_col:
        st.markdown(f"### {label}")
        st.markdown(
            f"**Score breakdown:** 100 "
            f"− {int(report['missing_spike'].sum())} missing spikes (×5) "
            f"− {int(report['range_issue'].sum())} range issues (×5) "
            f"− {int(report['drift_detected'].sum())} drifts (×10) "
            f"= **{score}/100**"
        )

    with counts_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("Missing Spikes", int(report["missing_spike"].sum()))
        m2.metric("Range Violations", int(report["range_issue"].sum()))
        m3.metric("Drift Detected", int(report["drift_detected"].sum()))

    # ---- Detailed report table ----
    st.markdown("---")
    st.subheader("📋 Feature-Level Report")

    display_report = report[
        [
            "feature",
            "missing_spike",
            "range_issue",
            "drift_detected",
            "ks_statistic",
            "p_value",
            "status",
        ]
    ].copy()
    display_report.columns = [
        "Feature",
        "Missing Spike",
        "Range Issue",
        "Drift Detected",
        "KS Statistic",
        "p-value",
        "Status",
    ]

    st.dataframe(
        display_report.style.applymap(
            lambda v: "background-color: #FFCDD2" if v == "⚠ Issue" else "",
            subset=["Status"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Issues bar chart ----
    st.plotly_chart(plot_feature_status_bar(report), use_container_width=True)

    # ---- Distribution comparisons ----
    st.markdown("---")
    st.subheader("📈 Distribution Comparison")

    selected_feature = st.selectbox(
        "Select a feature to compare distributions",
        shared_cols,
        index=shared_cols.index("MonthlyIncome")
        if "MonthlyIncome" in shared_cols
        else 0,
    )

    st.plotly_chart(
        plot_distribution_comparison(
            df_baseline[selected_feature],
            df_current[selected_feature],
            selected_feature,
        ),
        use_container_width=True,
    )

    # Show feature-level stats side by side
    with st.expander(f"📊 Detailed stats for **{selected_feature}**"):
        stat1, stat2 = st.columns(2)
        with stat1:
            st.markdown("**Baseline**")
            st.write(df_baseline[selected_feature].describe())
        with stat2:
            st.markdown("**Current**")
            st.write(df_current[selected_feature].describe())

    # ---- Metric explanations ----
    st.markdown("---")
    with st.expander("ℹ️ What do these metrics mean?", expanded=False):
        st.markdown(
            """
**Kolmogorov-Smirnov (KS) Test**
A non-parametric test that compares the empirical cumulative distribution
functions (CDFs) of two samples. The KS statistic measures the maximum
distance between the two CDFs. A p-value below 0.05 indicates statistically
significant evidence that the two distributions differ.

**Missing Value Spike**
Flagged when the percentage of missing values in the current dataset exceeds
the baseline's missing rate by more than 5 percentage points. Sudden spikes
in missingness often indicate upstream pipeline failures or schema changes.

**Range Violation**
Flagged when the current dataset contains values outside the baseline's
observed [min, max] range. Out-of-range values can cause model errors,
especially for tree-based models that never saw such values during training.

**Data Reliability Score (0–100)**
A composite health metric. Starts at 100 and deducts 5 points per missing
spike, 5 per range violation, and 10 per confirmed drift detection.
Scores ≥ 85 are considered safe; 60–84 warrant review; below 60 means the
data should not be used for ML predictions without investigation.
            """
        )

    # ---- Download report ----
    st.markdown("---")
    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Report as CSV",
        data=csv_bytes,
        file_name="drift_monitoring_report.csv",
        mime="text/csv",
    )

else:
    # Landing state — no data loaded yet
    st.markdown(
        """
        ### Welcome!

        This tool helps you detect **data quality issues** and **statistical
        distribution drift** between a baseline (training) dataset and new
        incoming data.

        **Get started →** use the sidebar to load the pre-built sample data
        or upload your own CSV files.

        ---

        **What it checks:**

        | Check | Method | Threshold |
        |-------|--------|-----------|
        | Missing-value spikes | Δ missing % vs. baseline | > 5 pp increase |
        | Range violations | Current min/max vs. baseline envelope | Any breach |
        | Distribution drift | Kolmogorov-Smirnov two-sample test | p < 0.05 |
        | Reliability Score | Composite deduction formula | 0–100 scale |
        """
    )
