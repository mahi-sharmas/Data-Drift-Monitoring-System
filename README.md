# Data Drift & Quality Monitoring System

A production-style data monitoring pipeline that detects distribution drift, data quality issues, and generates reliability scores for ML systems. Built to simulate real-world MLOps workflows where models degrade silently due to upstream data changes.

## Problem Statement

Machine learning models in production often fail not because of code bugs, but because the input data changes over time — a phenomenon known as **data drift**. This project builds an automated monitoring system that compares incoming data against a stored baseline profile and flags anomalies before they impact model performance.

## Dataset

- **Source:** Kaggle — Give Me Some Credit (credit risk dataset)
- **Baseline Size:** 150,000 records, 11 features
- **Features:** `RevolvingUtilizationOfUnsecuredLines`, `age`, `DebtRatio`, `MonthlyIncome`, `NumberOfTimes90DaysLate`, and 6 others
- **Simulated Drift:** Missing value injection (10%), income distribution scaling (1.5x), impossible age values (150)

## Approach

1. **Baseline Profiling** — Computed and stored per-feature statistics (mean, std, min, max, missing %) as a JSON reference profile
2. **Data Quality Checks** — Compared incoming data against baseline for missing value spikes (>5% increase) and out-of-range values
3. **Statistical Drift Detection** — Applied the Kolmogorov-Smirnov (KS) two-sample test per feature with p < 0.05 threshold
4. **Reliability Scoring** — Composite score (0–100) penalizing quality issues (-5 each) and drift detections (-10 each)
5. **Streamlit Dashboard** — Interactive web app for uploading baseline/today CSVs, viewing reports, and visualizing drift via KDE plots

## Key Results

| Check Type | Flagged Features | Details |
|---|---|---|
| Missing Value Spike | 3 / 11 | `SeriousDlqin2yrs`, `RevolvingUtilization`, `age` |
| Range Violation | 2 / 11 | `age` (max 150 vs baseline 109), `MonthlyIncome` (scaled 1.5x) |
| KS Drift Detected | 2 / 11 | `MonthlyIncome` (p ≈ 0), `age` (p ≈ 8e-31) |
| **Reliability Score** | **55 / 100** | Verdict: Data NOT reliable for ML or decision-making |

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, SciPy (KS test), Seaborn, Matplotlib
- **Dashboard:** Streamlit
- **Concepts:** Data drift detection, statistical testing, MLOps monitoring, data quality assurance

## Project Structure

```
├── Data_Drift_Monitoring_System.ipynb   # Full analysis notebook
├── app.py                               # Streamlit dashboard
├── baseline_profile.json                # Stored baseline statistics
├── data_monitoring_report.csv           # Final feature-level report
├── cs-training.csv                      # Baseline dataset
├── cs-test.csv                          # Test dataset
└── README.md
```

## How to Run

```bash
# Run the notebook
jupyter notebook Data_Drift_Monitoring_System.ipynb

# Launch the Streamlit dashboard
pip install streamlit scipy seaborn
streamlit run app.py
```

## Author

**Mahi Sharma**
B.Tech CSE (Data Science) — Manipal University Jaipur
[GitHub](https://github.com/mahi-sharmas)
