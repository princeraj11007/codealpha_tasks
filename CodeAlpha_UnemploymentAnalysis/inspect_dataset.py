"""
CodeAlpha Task 2: Dataset Verification & Inspection Script
Kaggle Dataset: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india
Primary CSV: 'Unemployment in India.csv'
Secondary CSV: 'Unemployment_Rate_upto_11_2020.csv'
"""

import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

def inspect_file(filename):
    csv_path = os.path.join(base_dir, 'data', filename)
    df = pd.read_csv(csv_path)

    print("=" * 75)
    print(f"INSPECTING: {filename}")
    print(f"Source: Kaggle Dataset (gokulrajkmv/unemployment-in-india)")
    print("=" * 75)

    # 1. Load Dataset
    print(f"\n[1] LOAD ACTUAL CSV FILE:\nLoaded from: {csv_path}")

    # 2. First 5 Rows
    print(f"\n[2] FIRST 5 ROWS:\n{df.head(5).to_string()}")

    # 3. Last 5 Rows
    print(f"\n[3] LAST 5 ROWS:\n{df.tail(5).to_string()}")

    # 4. Dataset Shape
    print(f"\n[4] DATASET SHAPE:\nTotal Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    # 5. All Column Names
    print(f"\n[5] ALL RAW COLUMN NAMES:")
    for i, col in enumerate(df.columns, 1):
        print(f"  Col {i}: '{col}' (has leading whitespace: {col.startswith(' ')})")

    # 6. Data Types
    print(f"\n[6] DATA TYPES:\n{df.dtypes.to_string()}")

    # 7. Missing Values
    print(f"\n[7] MISSING VALUES (NULL COUNT):\n{df.isnull().sum().to_string()}")

    # 8. Duplicate Rows
    print(f"\n[8] DUPLICATE ROWS:\nTotal Duplicates: {df.duplicated().sum()}")

    # 9. Date Column Identification
    date_col = [c for c in df.columns if 'date' in c.lower()][0]
    print(f"\n[9] DATE COLUMN IDENTIFIED:\nColumn Name: '{date_col}'")
    print(f"Sample raw values: {df[date_col].dropna().unique()[:4].tolist()}")

    # 10. Unemployment Rate Column Identification
    unemp_col = [c for c in df.columns if 'unemployment' in c.lower()][0]
    print(f"\n[10] UNEMPLOYMENT RATE COLUMN IDENTIFIED:\nColumn Name: '{unemp_col}'")
    print(f"Value Type: {df[unemp_col].dtype}")
    print(f"Sample values: {df[unemp_col].dropna().head(5).tolist()}")

    # 11. Region / State Related Columns
    region_cols = [c for c in df.columns if any(k in c.lower() for k in ['region', 'state', 'area', 'zone'])]
    print(f"\n[11] REGION / STATE / GEOGRAPHIC COLUMNS IDENTIFIED:\nFound Columns: {region_cols}")
    for col in region_cols:
        unique_vals = df[col].dropna().unique().tolist()
        print(f" - '{col}': {len(unique_vals)} unique values (Samples: {unique_vals[:5]})")

    # 12. Actual Date Range Covered
    cleaned_dates = pd.to_datetime(df[date_col].dropna().str.strip(), format='%d-%m-%Y')
    print(f"\n[12] ACTUAL DATE RANGE COVERED BY DATASET:")
    print(f" - Earliest Date: {cleaned_dates.min().strftime('%d %B %Y')}")
    print(f" - Latest Date:   {cleaned_dates.max().strftime('%d %B %Y')}")
    print(f" - Total Unique Months: {cleaned_dates.nunique()} months")
    print(f" - Monthly Timeline: {[d.strftime('%b %Y') for d in sorted(cleaned_dates.unique())]}")
    print("\n")

inspect_file('Unemployment in India.csv')
inspect_file('Unemployment_Rate_upto_11_2020.csv')
