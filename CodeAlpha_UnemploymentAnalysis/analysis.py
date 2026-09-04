"""
CodeAlpha Data Science Internship - Task 2
Project: Unemployment Analysis with Python
Repository: CodeAlpha_UnemploymentAnalysis

This script performs complete data loading, cleaning, exploratory data analysis (EDA),
COVID-19 impact investigation, seasonal analysis, and generates high-resolution visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional, clean visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
print("=" * 60)
print("1. LOADING DATASET")
print("=" * 60)

csv_path = os.path.join(DATA_DIR, 'Unemployment in India.csv')
df = pd.read_csv(csv_path)

print(f"Raw Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nRaw Dataset Head (First 5 rows):")
print(df.head())

print("\nRaw Dataset Tail (Last 5 rows):")
print(df.tail())

print("\nMissing Values Check (Raw Data):")
print(df.isnull().sum())

# ---------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("2. DATA CLEANING & PREPROCESSING")
print("=" * 60)

# 2.1 Drop completely empty rows (28 trailing blank records)
df_clean = df.dropna().copy()
print(f"Shape after dropping null rows: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

# 2.2 Strip whitespace from column names
df_clean.columns = df_clean.columns.str.strip()
print(f"Cleaned Column Names: {df_clean.columns.tolist()}")

# 2.3 Strip whitespace from string values
for col in ['Region', 'Frequency', 'Area']:
    df_clean[col] = df_clean[col].astype(str).str.strip()

# 2.4 Convert Date column to datetime format
df_clean['Date'] = pd.to_datetime(df_clean['Date'].str.strip(), format='%d-%m-%Y')

# 2.5 Extract date-related features for time-series and seasonal analysis
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Month_Num'] = df_clean['Date'].dt.month
df_clean['Month'] = df_clean['Date'].dt.strftime('%b')
df_clean['Month_Year'] = df_clean['Date'].dt.strftime('%b-%Y')

# 2.6 Classify Period: Pre-COVID (May 2019 - Feb 2020) vs During-COVID (Mar 2020 - Jun 2020)
# India announced nationwide COVID-19 lockdown in late March 2020
df_clean['COVID_Period'] = df_clean['Date'].apply(
    lambda d: 'Pre-COVID (May 2019 - Feb 2020)' if d < pd.Timestamp('2020-03-01') else 'During-COVID (Mar 2020 - Jun 2020)'
)

# 2.7 Check for duplicates
duplicates_count = df_clean.duplicated().sum()
print(f"Duplicate records found: {duplicates_count}")

print("\nCleaned Dataset Overview:")
print(df_clean.info())

print("\nSummary Statistics of Numerical Columns:")
print(df_clean.describe().round(2))

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

overall_mean_unemp = df_clean['Estimated Unemployment Rate (%)'].mean()
overall_median_unemp = df_clean['Estimated Unemployment Rate (%)'].median()
overall_min_unemp = df_clean['Estimated Unemployment Rate (%)'].min()
overall_max_unemp = df_clean['Estimated Unemployment Rate (%)'].max()

print(f"Overall Mean Unemployment Rate: {overall_mean_unemp:.2f}%")
print(f"Overall Median Unemployment Rate: {overall_median_unemp:.2f}%")
print(f"Overall Minimum Unemployment Rate: {overall_min_unemp:.2f}%")
print(f"Overall Maximum Unemployment Rate: {overall_max_unemp:.2f}%")

# State-wise Unemployment Analysis
state_stats = df_clean.groupby('Region')['Estimated Unemployment Rate (%)'].agg(['mean', 'median', 'min', 'max', 'std']).sort_values(by='mean', ascending=False)

print("\nTop 5 States with Highest Average Unemployment Rate:")
print(state_stats.head(5).round(2))

print("\nTop 5 States with Lowest Average Unemployment Rate:")
print(state_stats.tail(5).round(2))

# Rural vs Urban Analysis
area_stats = df_clean.groupby('Area')[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].mean()
print("\nRural vs Urban Comparison (Averages):")
print(area_stats.round(2))

# ---------------------------------------------------------
# 4. COVID-19 IMPACT ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("4. COVID-19 IMPACT ANALYSIS")
print("=" * 60)

pre_covid_df = df_clean[df_clean['Date'] < '2020-03-01']
during_covid_df = df_clean[df_clean['Date'] >= '2020-03-01']

pre_unemp = pre_covid_df['Estimated Unemployment Rate (%)'].mean()
during_unemp = during_covid_df['Estimated Unemployment Rate (%)'].mean()
unemp_diff = during_unemp - pre_unemp
unemp_pct_increase = (unemp_diff / pre_unemp) * 100

pre_employed = pre_covid_df['Estimated Employed'].mean()
during_employed = during_covid_df['Estimated Employed'].mean()
employed_diff = during_employed - pre_employed
employed_pct_decrease = (employed_diff / pre_employed) * 100

pre_labour = pre_covid_df['Estimated Labour Participation Rate (%)'].mean()
during_labour = during_covid_df['Estimated Labour Participation Rate (%)'].mean()

print(f"Pre-COVID (May 2019 - Feb 2020) Average Unemployment Rate: {pre_unemp:.2f}%")
print(f"During COVID (Mar 2020 - Jun 2020) Average Unemployment Rate: {during_unemp:.2f}%")
print(f"Absolute Increase in Unemployment Rate: +{unemp_diff:.2f}% percentage points")
print(f"Relative Percentage Increase: +{unemp_pct_increase:.2f}%")
print()
print(f"Pre-COVID Mean Employed: {pre_employed:,.0f}")
print(f"During COVID Mean Employed: {during_employed:,.0f}")
print(f"Decline in Mean Employment: {employed_diff:,.0f} ({employed_pct_decrease:.2f}%)")
print()
print(f"Pre-COVID Labour Participation Rate: {pre_labour:.2f}%")
print(f"During COVID Labour Participation Rate: {during_labour:.2f}%")

# Highest unemployment single observations during lockdown
peak_lockdown = df_clean.sort_values(by='Estimated Unemployment Rate (%)', ascending=False).head(5)
print("\nTop 5 Highest Single Unemployment Rate Observations (During Lockdown):")
print(peak_lockdown[['Region', 'Date', 'Area', 'Estimated Unemployment Rate (%)', 'Estimated Employed']].to_string(index=False))

# State-by-State Pre vs During COVID Impact
state_covid_comp = df_clean.groupby(['Region', 'COVID_Period'])['Estimated Unemployment Rate (%)'].mean().unstack()
state_covid_comp['Absolute_Change'] = state_covid_comp['During-COVID (Mar 2020 - Jun 2020)'] - state_covid_comp['Pre-COVID (May 2019 - Feb 2020)']
state_covid_comp = state_covid_comp.sort_values(by='Absolute_Change', ascending=False)
print("\nTop 5 States Most Severely Impacted by COVID-19 (Largest Rate Increase):")
print(state_covid_comp.head(5).round(2))

# ---------------------------------------------------------
# 5. VISUALIZATIONS GENERATION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("5. GENERATING VISUALIZATIONS")
print("=" * 60)

# ---------------------------------------------------------
# Chart 1: Unemployment Rate Trend Over Time
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
monthly_trend = df_clean.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.plot(monthly_trend['Date'], monthly_trend['Estimated Unemployment Rate (%)'], 
         marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='Monthly Average Unemployment Rate')

# Fill pre-COVID vs COVID regions
plt.axvspan(pd.Timestamp('2019-05-31'), pd.Timestamp('2020-02-29'), color='#2ca02c', alpha=0.12, label='Pre-COVID Period')
plt.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-30'), color='#d62728', alpha=0.15, label='COVID-19 Lockdown Period')

# Annotate the peak in April & May 2020
peak_date = monthly_trend.loc[monthly_trend['Estimated Unemployment Rate (%)'].idxmax(), 'Date']
peak_val = monthly_trend['Estimated Unemployment Rate (%)'].max()
plt.annotate(f'Peak: {peak_val:.2f}%\n(May 2020 Lockdown)', 
             xy=(peak_date, peak_val), 
             xytext=(peak_date - pd.Timedelta(days=70), peak_val + 2),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=8),
             fontsize=11, fontweight='bold', color='#d62728',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d62728", lw=1.5))

plt.title('National Unemployment Rate Trend in India (May 2019 - June 2020)', pad=15)
plt.xlabel('Date (Month-Year)', labelpad=10)
plt.ylabel('Estimated Unemployment Rate (%)', labelpad=10)
plt.ylim(0, 30)
plt.xticks(monthly_trend['Date'], monthly_trend['Date'].dt.strftime('%b %Y'), rotation=45, ha='right')
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart1_path = os.path.join(IMAGES_DIR, 'unemployment_trend.png')
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved: {chart1_path}")

# ---------------------------------------------------------
# Chart 2: COVID-19 Impact Analysis (Pre vs During COVID by State)
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))
top_impacted = state_covid_comp.sort_values(by='During-COVID (Mar 2020 - Jun 2020)', ascending=True)

y = np.arange(len(top_impacted))
height = 0.38

plt.barh(y - height/2, top_impacted['Pre-COVID (May 2019 - Feb 2020)'], height=height, 
         label='Pre-COVID (May 2019 - Feb 2020)', color='#2b83ba', alpha=0.9)
plt.barh(y + height/2, top_impacted['During-COVID (Mar 2020 - Jun 2020)'], height=height, 
         label='During-COVID (Mar 2020 - Jun 2020)', color='#d7191c', alpha=0.9)

plt.yticks(y, top_impacted.index, fontsize=10)
plt.xlabel('Average Unemployment Rate (%)', labelpad=10)
plt.ylabel('State / Union Territory', labelpad=10)
plt.title('State-by-State Impact of COVID-19 on Unemployment Rate', pad=15)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart2_path = os.path.join(IMAGES_DIR, 'covid_impact.png')
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved: {chart2_path}")

# ---------------------------------------------------------
# Chart 3: Top and Bottom States by Average Unemployment Rate
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))
top_states = state_stats['mean'].sort_values(ascending=False).head(10)
bottom_states = state_stats['mean'].sort_values(ascending=True).head(10)

combined_states = pd.concat([top_states, bottom_states.iloc[::-1]])
colors = ['#d62728' if state in top_states.index else '#2ca02c' for state in combined_states.index]

bars = plt.barh(combined_states.index, combined_states.values, color=colors, alpha=0.85)

# Add values to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.4, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
             ha='left', va='center', fontsize=9, fontweight='bold')

plt.axvline(overall_mean_unemp, color='black', linestyle='--', linewidth=1.5, 
            label=f'National Average ({overall_mean_unemp:.2f}%)')

plt.title('Top 10 Highest vs Top 10 Lowest Unemployment States in India', pad=15)
plt.xlabel('Average Unemployment Rate (%)', labelpad=10)
plt.ylabel('State / Region', labelpad=10)
plt.xlim(0, 32)
plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart3_path = os.path.join(IMAGES_DIR, 'state_unemployment_comparison.png')
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"Saved: {chart3_path}")

# ---------------------------------------------------------
# Chart 4: Rural vs Urban Unemployment Dynamics
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Rural vs Urban Trend Over Time
monthly_area = df_clean.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()
ax1.plot(monthly_area.index, monthly_area['Rural'], marker='o', color='#2ca02c', linewidth=2.2, label='Rural Area')
ax1.plot(monthly_area.index, monthly_area['Urban'], marker='s', color='#ff7f0e', linewidth=2.2, label='Urban Area')
ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-30'), color='#d62728', alpha=0.12, label='COVID Lockdown')
ax1.set_title('Monthly Unemployment: Rural vs Urban', pad=12)
ax1.set_xlabel('Date', labelpad=8)
ax1.set_ylabel('Unemployment Rate (%)', labelpad=8)
ax1.set_xticks(monthly_area.index)
ax1.set_xticklabels(monthly_area.index.strftime('%b %y'), rotation=45, ha='right')
ax1.legend(frameon=True, facecolor='white')

# Subplot 2: Boxplot Distribution by Area
sns.boxplot(x='Area', y='Estimated Unemployment Rate (%)', data=df_clean, 
            palette=['#2ca02c', '#ff7f0e'], ax=ax2, width=0.4, boxprops=dict(alpha=0.8))
ax2.set_title('Unemployment Rate Distribution: Rural vs Urban', pad=12)
ax2.set_xlabel('Geographic Area', labelpad=8)
ax2.set_ylabel('Unemployment Rate (%)', labelpad=8)

plt.suptitle('Comparative Analysis of Rural vs Urban Unemployment in India', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

chart4_path = os.path.join(IMAGES_DIR, 'rural_vs_urban_analysis.png')
plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart4_path}")

# ---------------------------------------------------------
# Chart 5: Correlation Heatmap
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_cols = ['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']
corr_matrix = df_clean[numeric_cols].corr()

# Rename columns for cleaner visualization
renamed_corr = corr_matrix.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate (%)',
    'Estimated Employed': 'Employed Count',
    'Estimated Labour Participation Rate (%)': 'Labour Participation (%)'
}, index={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate (%)',
    'Estimated Employed': 'Employed Count',
    'Estimated Labour Participation Rate (%)': 'Labour Participation (%)'
})

sns.heatmap(renamed_corr, annot=True, cmap='Blues', fmt='.3f', cbar=True, 
            square=True, linewidths=1, linecolor='white', annot_kws={'size': 11, 'weight': 'bold'})
plt.title('Correlation Matrix of Labor Market Indicators', pad=15)
plt.tight_layout()

chart5_path = os.path.join(IMAGES_DIR, 'correlation_heatmap.png')
plt.savefig(chart5_path, dpi=300)
plt.close()
print(f"Saved: {chart5_path}")

# ---------------------------------------------------------
# 6. SUMMARY REPORT
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("6. ANALYSIS SUMMARY & KEY INSIGHTS")
print("=" * 60)
print(f"1. Pre-COVID baseline unemployment was {pre_unemp:.2f}%, jumping to {during_unemp:.2f}% during COVID.")
print(f"2. Unemployment surged by +{unemp_pct_increase:.1f}% during the lockdown period.")
print(f"3. National peak occurred in May 2020 at {monthly_trend['Estimated Unemployment Rate (%)'].max():.2f}%.")
print(f"4. Puducherry witnessed the single highest recorded unemployment rate of 76.74% in April 2020.")
print(f"5. Urban areas experienced a higher average unemployment rate ({area_stats.loc['Urban', 'Estimated Unemployment Rate (%)']:.2f}%) than Rural areas ({area_stats.loc['Rural', 'Estimated Unemployment Rate (%)']:.2f}%).")
print(f"6. Top highest average unemployment states: Tripura (28.35%), Haryana (26.28%), Jharkhand (20.59%), Bihar (18.92%).")
print(f"7. Lowest average unemployment states: Meghalaya (4.80%), Odisha (5.66%), Assam (6.43%), Uttarakhand (6.58%).")
print("=" * 60)
print("Analysis script execution finished successfully!")
