"""
data_loader.py
--------------
Handles CSV loading, validation, and baseline statistical profiling.

Functions:
    load_csv          – Read a CSV from a file path or Streamlit UploadedFile
    validate_columns  – Check that two DataFrames share the same numeric columns
    build_baseline_profile – Compute per-column statistics for the baseline dataset
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
from io import StringIO


def load_csv(source: Union[str, "UploadedFile"]) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame and clean up common artifacts.

    Handles both local file paths (str) and Streamlit UploadedFile objects.
    Automatically drops Kaggle's auto-generated 'Unnamed: 0' index column.

    Parameters
    ----------
    source : str or UploadedFile
        Path to a CSV file, or a Streamlit file-upload object.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with only meaningful columns.
    """
    df = pd.read_csv(source)
    # Kaggle datasets often include an unnamed row-index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def validate_columns(
    baseline: pd.DataFrame, current: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Identify shared numeric columns and flag mismatches between two DataFrames.

    Parameters
    ----------
    baseline : pd.DataFrame
        The reference / training dataset.
    current : pd.DataFrame
        The incoming / production dataset to compare against.

    Returns
    -------
    shared_cols : list[str]
        Numeric columns present in both DataFrames.
    missing_cols : list[str]
        Numeric columns that exist in the baseline but are absent in current.
    """
    baseline_numeric = set(baseline.select_dtypes(include=[np.number]).columns)
    current_numeric = set(current.select_dtypes(include=[np.number]).columns)
    shared = sorted(baseline_numeric & current_numeric)
    missing = sorted(baseline_numeric - current_numeric)
    return shared, missing


def build_baseline_profile(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute a statistical profile for every numeric column in the DataFrame.

    The profile includes: mean, std, min, max, median, and missing_pct.
    This serves as the reference snapshot that incoming data is compared against.

    Parameters
    ----------
    df : pd.DataFrame
        The baseline / training dataset.

    Returns
    -------
    dict
        Nested dictionary keyed by column name → stat name → value.
        Example: {"age": {"mean": 52.3, "std": 14.8, ...}}
    """
    profile = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        profile[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "missing_pct": float(df[col].isna().mean()),
        }
    return profile
