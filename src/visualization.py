"""
visualization.py
----------------
Plotly-based interactive charts for the Streamlit dashboard.

Uses Plotly instead of Matplotlib/Seaborn for a richer interactive experience
inside Streamlit (hover tooltips, zoom, pan) without needing static renders.

Functions:
    plot_distribution_comparison – Overlaid KDE histograms (baseline vs. current)
    plot_reliability_gauge      – Gauge chart for the composite score
    plot_feature_status_bar     – Horizontal bar showing issue counts per feature
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_distribution_comparison(
    baseline_col: pd.Series,
    current_col: pd.Series,
    feature_name: str,
) -> go.Figure:
    """
    Create an overlaid histogram with KDE curves comparing baseline vs. current.

    Parameters
    ----------
    baseline_col : pd.Series
        Baseline feature values (NaN dropped internally).
    current_col : pd.Series
        Current feature values (NaN dropped internally).
    feature_name : str
        Name of the feature (used in title and legend).

    Returns
    -------
    go.Figure
        A Plotly figure ready for st.plotly_chart().
    """
    base_clean = baseline_col.dropna()
    curr_clean = current_col.dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=base_clean,
            name="Baseline",
            opacity=0.55,
            marker_color="#636EFA",
            histnorm="probability density",
            nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=curr_clean,
            name="Current",
            opacity=0.55,
            marker_color="#EF553B",
            histnorm="probability density",
            nbinsx=60,
        )
    )
    fig.update_layout(
        title=dict(text=f"Distribution Comparison: {feature_name}", font_size=16),
        xaxis_title=feature_name,
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        height=400,
        margin=dict(t=50, b=40, l=50, r=30),
    )
    return fig


def plot_reliability_gauge(score: int, label: str) -> go.Figure:
    """
    Render a gauge-chart for the composite Data Reliability Score.

    Colour bands: 0-59 red, 60-84 orange, 85-100 green.

    Parameters
    ----------
    score : int
        Reliability score (0–100).
    label : str
        Classification text (e.g. "Safe for ML").

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Data Reliability Score", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 60], "color": "#FFCDD2"},
                    {"range": [60, 85], "color": "#FFF9C4"},
                    {"range": [85, 100], "color": "#C8E6C9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=40, r=40))
    return fig


def plot_feature_status_bar(report: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked bar chart showing which checks each feature failed.

    Parameters
    ----------
    report : pd.DataFrame
        Merged report from run_full_analysis().

    Returns
    -------
    go.Figure
    """
    df = report[["feature", "missing_spike", "range_issue", "drift_detected"]].copy()
    df["missing_spike"] = df["missing_spike"].astype(int)
    df["range_issue"] = df["range_issue"].astype(int)
    df["drift_detected"] = df["drift_detected"].astype(int)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["feature"],
            x=df["missing_spike"],
            name="Missing Spike",
            orientation="h",
            marker_color="#EF553B",
        )
    )
    fig.add_trace(
        go.Bar(
            y=df["feature"],
            x=df["range_issue"],
            name="Range Violation",
            orientation="h",
            marker_color="#FFA15A",
        )
    )
    fig.add_trace(
        go.Bar(
            y=df["feature"],
            x=df["drift_detected"],
            name="Drift Detected",
            orientation="h",
            marker_color="#AB63FA",
        )
    )
    fig.update_layout(
        barmode="stack",
        title=dict(text="Issues by Feature", font_size=16),
        xaxis_title="Number of Issues",
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(df) * 35),
        margin=dict(t=50, b=40, l=10, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
