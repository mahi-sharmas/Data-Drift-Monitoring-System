import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Data Drift & Quality Monitoring Dashboard")

# Upload baseline and today data
baseline_file = st.file_uploader("Upload Baseline Data", type=["csv"])
today_file = st.file_uploader("Upload Today's Data", type=["csv"])

if baseline_file and today_file:
    # Drop the Kaggle auto index if present
    df_baseline = pd.read_csv(baseline_file).drop(columns=['Unnamed: 0'], errors='ignore')
    df_today = pd.read_csv(today_file).drop(columns=['Unnamed: 0'], errors='ignore')

    st.subheader("Baseline Data Preview")
    st.write(df_baseline.head())

    st.subheader("Today's Data Preview")
    st.write(df_today.head())


    results = []

    for col in df_baseline.columns:
        if df_baseline[col].dtype != 'object':
            # Missing values
            missing_spike = df_today[col].isna().mean() - df_baseline[col].isna().mean() > 0.05

            # Range check
            range_issue = (
                df_today[col].min() < df_baseline[col].min() or
                df_today[col].max() > df_baseline[col].max()
            )

            # Drift test
            stat, p_value = ks_2samp(df_baseline[col].dropna(), df_today[col].dropna())
            drift_detected = p_value < 0.05

            results.append({
                "feature": col,
                "missing_spike": missing_spike,
                "range_issue": range_issue,
                "drift_detected": drift_detected,
                "p_value": p_value
            })

    report = pd.DataFrame(results)

    st.subheader("Monitoring Report")
    st.dataframe(report)

    # Plot drift for selected feature
    feature = st.selectbox("Select Feature to Visualize Drift", df_baseline.columns)

    fig, ax = plt.subplots()
    sns.kdeplot(df_baseline[feature], label='Baseline', fill=True, ax=ax)
    sns.kdeplot(df_today[feature], label='Today', fill=True, ax=ax)
    ax.set_title(f'Distribution Drift: {feature}')
    ax.legend()
    st.pyplot(fig)
