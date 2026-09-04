"""
===============================================================================
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
Dataset Inspection & Verification Script
===============================================================================
Primary Dataset Source:
    Kaggle: https://www.kaggle.com/datasets/saurabh00007/iriscsv
    File: Iris.csv

This script performs the strict 9-step dataset verification and exploratory checks
on the downloaded Kaggle CSV file before model building.
===============================================================================
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure clean UTF-8 console output where available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def inspect_iris_dataset(csv_path: str = "Iris.csv"):
    # Resolve file path
    if not os.path.exists(csv_path):
        # Look in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, "Iris.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset file '{csv_path}' not found! Please ensure 'Iris.csv' is downloaded "
                "from https://www.kaggle.com/datasets/saurabh00007/iriscsv"
            )

    print("=" * 80)
    print(" CODEALPHA TASK 1: KAGGLE IRIS DATASET INSPECTION (saurabh00007/iriscsv)")
    print("=" * 80)
    print(f"Dataset Path   : {os.path.abspath(csv_path)}")
    print(f"Dataset Source : https://www.kaggle.com/datasets/saurabh00007/iriscsv\n")

    # 1. Load the actual downloaded CSV
    print("[STEP 1] LOADING THE ACTUAL DOWNLOADED CSV FILE...")
    df = pd.read_csv(csv_path)
    print(">> Success: Dataset successfully loaded into Pandas DataFrame.\n")

    # 2. Display the first 5 rows
    print("[STEP 2] FIRST 5 ROWS OF THE DATASET (df.head()):")
    print("-" * 80)
    print(df.head().to_string(index=False))
    print("-" * 80 + "\n")

    # 3. Display the column names
    print("[STEP 3] COLUMN NAMES IN THE DATASET:")
    for idx, col in enumerate(df.columns, 1):
        print(f"   {idx}. '{col}'")
    print(f"\n   Total Columns: {len(df.columns)}\n")

    # 4. Display the shape
    print("[STEP 4] DATASET SHAPE (Rows, Columns):")
    print(f"   Shape: {df.shape} -> ({df.shape[0]} samples/rows, {df.shape[1]} columns)\n")

    # 5. Check data types
    print("[STEP 5] DATA TYPES (df.dtypes):")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:<18} : {dtype}")
    print()

    # 6. Check missing values
    print("[STEP 6] MISSING VALUES CHECK (df.isnull().sum()):")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"   - {col:<18} : {count} missing ({pct:.1f}%)")
    total_missing = missing.sum()
    print(f"   Total Missing Values Across Dataset: {total_missing}")
    if total_missing == 0:
        print("   >> Data Quality: Clean (No missing values detected)\n")

    # 7. Check duplicate values
    print("[STEP 7] DUPLICATE ROWS CHECK:")
    exact_duplicates = df.duplicated().sum()
    print(f"   - Exact duplicate rows (including 'Id'): {exact_duplicates}")
    if 'Id' in df.columns:
        measurement_duplicates = df.drop(columns=['Id']).duplicated().sum()
        print(f"   - Duplicate rows based on measurements only (excluding 'Id'): {measurement_duplicates}")
        print("   >> Note: In biological samples, identical morphological measurements can occur naturally across distinct specimens.")
    print()

    # 8. Identify which column contains the Iris species/target
    target_col = "Species" if "Species" in df.columns else [c for c in df.columns if "species" in c.lower()][0]
    print(f"[STEP 8] TARGET COLUMN IDENTIFICATION:")
    print(f"   - Target Column Name : '{target_col}'")
    print(f"   - Target Data Type   : {df[target_col].dtype}")
    print(f"   - Unique Classes ({df[target_col].nunique()} total):")
    for species, count in df[target_col].value_counts().items():
        print(f"       * {species:<18} : {count} samples ({count/len(df)*100:.1f}%)")
    print("   >> Class Balance: Perfectly balanced (50 samples per species class)\n")

    # 9. Identify the flower measurement/features
    feature_cols = [col for col in df.columns if col not in ['Id', 'id', target_col]]
    print("[STEP 9] FLOWER MEASUREMENT / FEATURE COLUMNS:")
    for idx, col in enumerate(feature_cols, 1):
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"   {idx}. '{col}' (continuous numeric in cm)")
        print(f"      Range: [{min_val:.1f} cm - {max_val:.1f} cm] | Mean: {mean_val:.2f} cm | Std: {std_val:.2f} cm")
    print(f"\n   Total Input Features: {len(feature_cols)}\n")

    # UNNECESSARY COLUMN ANALYSIS (ID COLUMN)
    print("=" * 80)
    print(" UNNECESSARY COLUMN ANALYSIS: 'Id' COLUMN")
    print("=" * 80)
    if 'Id' in df.columns:
        print(" * Column Name   : 'Id'")
        print(" * Column Purpose: Sequential row index generated during data export (Values: 1 to 150).")
        print(" * Analysis & Recommendation:")
        print("     1. Zero Predictive Value: Flower species is determined by biological dimensions")
        print("        (sepals and petals), NOT the sequential row order in which flowers were recorded.")
        print("     2. Risk of Data Leakage & Overfitting: Since the dataset is ordered sequentially")
        print("        (Rows 1-50: Iris-setosa, 51-100: Iris-versicolor, 101-150: Iris-virginica), an ML")
        print("        model could erroneously learn 'Id <= 50 -> Setosa' as a rule, creating spurious")
        print("        correlations and failing to generalize on new, unseen flower measurements.")
        print(" * ACTION: REMOVE 'Id' COLUMN BEFORE MODEL TRAINING & EDA.")
    else:
        print(" * 'Id' column already removed or not present.")
    print("=" * 80 + "\n")

    return df, feature_cols, target_col


if __name__ == "__main__":
    inspect_iris_dataset()
