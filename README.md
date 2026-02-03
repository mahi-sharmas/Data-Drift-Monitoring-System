# Data Drift & Quality Monitoring System

A Python-based system to monitor data quality and detect data drift in incoming datasets, ensuring reliable analytics and machine learning pipelines. Designed for real-world production scenarios like fintech, e-commerce, and health tech.

## Features

- **Data Validation Checks**  
  - Detects missing value spikes, schema mismatches, out-of-range values.
- **Data Drift Detection**  
  - Uses Kolmogorov–Smirnov (KS) test for numerical features and Chi-Square test for categorical features.
- **Data Reliability Score**  
  - Computes an overall score for dataset health and flags columns that may break downstream analytics or ML models.
- **Interactive Dashboard (Streamlit)**  
  - Upload baseline & incoming datasets, view drift plots, and export monitoring reports.

## How to Run

1. Install dependencies:

```bash

pip install pandas numpy scipy matplotlib seaborn streamlit


Run the Streamlit dashboard:
streamlit run app.py


Files in this Repository

Data_Drift_Monitoring_System.ipynb → Jupyter/Colab notebook with full analysis
app.py → Streamlit dashboard
cs-training.csv → Baseline dataset
today_data.csv → Example incoming dataset
baseline_profile.json → Precomputed baseline statistics
data_monitoring_report.csv → Sample monitoring report
