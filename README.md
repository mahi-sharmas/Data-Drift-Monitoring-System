## Data Drift Monitoring System

A real-time data quality and statistical drift detection pipeline with an interactive Streamlit dashboard, built to monitor ML model reliability in production environments.

### 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-drift-monitor.streamlit.app)

> **Try it now** — the app loads with pre-built sample data so you can explore every feature instantly, or upload your own CSV to test against.

### Highlights

- Automated detection of missing value spikes, range violations, and distribution drift using the Kolmogorov-Smirnov test across 11 features
- Composite Data Reliability Score (0–100) that classifies datasets as Safe (≥85), Review Needed (60–84), or Unreliable (<60) for ML
- Interactive Streamlit dashboard with Plotly visualizations, file upload, and downloadable CSV reports
- Modular, production-ready codebase with separated data loading, drift detection, and visualization layers
- Tested on a 150,000-record credit default dataset with synthetically injected drift — correctly flagged all 3 drift types and scored 55/100

### Problem Statement

Machine learning models degrade silently when the data they consume drifts from what they were trained on. Traditional monitoring catches code errors but misses subtle data quality issues — missing value spikes, distribution shifts, and out-of-range anomalies. This project builds an automated monitoring system that profiles baseline data, statistically tests incoming data for drift, and assigns a composite reliability score so teams know whether their data is safe for ML or decision-making.

### Dataset

- **Source:** Kaggle — Give Me Some Credit (Credit Default Dataset)
- **Size:** 150,000 rows × 11 numeric features
- **Target:** `SeriousDlqin2yrs` (binary: serious delinquency in 2 years)
- **Key features:** RevolvingUtilizationOfUnsecuredLines, age (mean 52.3), DebtRatio, MonthlyIncome (mean $6,670, 19.8% missing), NumberOfDependents (2.6% missing), and 5 delinquency count features

### Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange)
![SciPy](https://img.shields.io/badge/SciPy-1.11+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75)

### Methodology

1. **Baseline Profiling** — Computed statistical profiles (mean, std, min, max, missing %) for all 11 features from the training dataset and saved to a JSON reference file
2. **Drift Injection (Testing)** — Simulated real-world issues: injected 10% NaN spikes in 3 columns, multiplied MonthlyIncome by 1.5×, and replaced 2% of age values with an impossible value (150)
3. **Quality Checks** — Detected missing value spikes (>5% increase over baseline) and range boundary violations (values exceeding baseline min/max)
4. **Statistical Testing** — Applied the Kolmogorov-Smirnov two-sample test (p < 0.05 threshold) to detect distribution shifts across all 11 features
5. **Reliability Scoring** — Computed a composite score starting at 100, deducting 5 points per missing spike, 5 per range issue, and 10 per confirmed drift detection
6. **Classification** — Categorized data as Safe (≥85), Review Needed (60–84), or Unreliable (<60) for ML use
7. **Dashboard Deployment** — Built a modular Streamlit app with file upload, automated analysis, interactive Plotly visualizations, and a downloadable monitoring report

### Key Results

| Check Type | Features Flagged | Details |
|---|---|---|
| Missing Value Spikes | 3 / 11 | SeriousDlqin2yrs, RevolvingUtilization, age |
| Range Violations | 2 / 11 | age (max 150 vs. baseline 109), MonthlyIncome (1.5× mean shift) |
| KS-Test Drift Detected | 2 / 11 | MonthlyIncome (p = 0.0), age (p = 7.98e-31) |
| **Data Reliability Score** | **55 / 100** | **❌ Data NOT reliable for ML or decision-making** |

**Score breakdown:** 100 − 15 (3 missing spikes × 5) − 10 (2 range issues × 5) − 20 (2 drifts × 10) = **55/100**

The system successfully detected all 3 injected drift types and correctly classified the degraded dataset as unreliable.

### How to Run

```bash
git clone https://github.com/mahi-sharmas/Data-Drift-Monitoring-System.git
cd Data-Drift-Monitoring-System
pip install -r requirements.txt
```

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Or explore the original notebook:

```bash
jupyter notebook Data_Drift_Monitoring_System.ipynb
```

### Project Structure

```
Data-Drift-Monitoring-System/
├── app.py                              # Streamlit dashboard — main entrypoint
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # CSV loading, validation, baseline profiling
│   ├── drift_detection.py              # KS tests, quality checks, reliability scoring
│   └── visualization.py                # Plotly charts (distributions, gauge, bar)
├── data/
│   ├── sample_baseline.csv             # 5K-row sample baseline (credit scoring)
│   └── sample_today.csv                # 5K-row sample with injected drift
├── .streamlit/
│   └── config.toml                     # Theme and server configuration
├── Data_Drift_Monitoring_System.ipynb   # Original analysis notebook
├── baseline_profile.json               # Saved baseline statistics (all 11 features)
├── data_monitoring_report.csv           # Generated drift report
├── cs-training.csv                      # Full baseline dataset (150K rows)
├── today_data.csv                       # Full test dataset with injected drift
├── requirements.txt                     # Python dependencies (pinned ranges)
└── README.md
```

### Future Improvements

- Add automated email/Slack alerts when the reliability score drops below a configurable threshold
- Integrate Population Stability Index (PSI) and Jensen-Shannon divergence as additional drift metrics alongside KS
- Connect to live data pipelines (e.g., Kafka or Airflow) for scheduled, production-grade monitoring

### Author

**Mahi Sharma** — B.Tech CSE (Data Science), Manipal University Jaipur (2023–2027)

GitHub: [github.com/mahi-sharmas](https://github.com/mahi-sharmas) | Email: mahi.sh4rma7@gmail.com

*Project completed: Jun 2025*
