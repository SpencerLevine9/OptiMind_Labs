"""
data_profile.py

Data Integrity & CSV Schema Owner
Dataset Profile, Schema Validation, Missing Values, Outlier Check

Compares both trajectory datasets side by side and exports a summary table
to results/dataset_profile.csv for use in the report.
"""

import os
import pandas as pd
import numpy as np

RANDOM_PATH = "data/random_trajectories.csv"
PPO_PATH    = "data/trajectories.csv"
OUTPUT_PATH = "results/dataset_profile.csv"

random_df = pd.read_csv(RANDOM_PATH)
ppo_df    = pd.read_csv(PPO_PATH)

print("=" * 60)
print("OPTIMIND LABS — DATASET PROFILE REPORT")
print("=" * 60)


# Schema Validation
# We check whether both CSVs share the same column names and data types.
# Since the random dataset has extra bookkeeping columns (episode_id, t) that
# the PPO dataset does not, we identify the shared "core" columns and validate
# those. Any mismatch is surfaced clearly so it can be documented in the report.

print("\n[1] SCHEMA VALIDATION")
print("-" * 40)

random_cols = set(random_df.columns)
ppo_cols    = set(ppo_df.columns)

only_in_random = random_cols - ppo_cols
only_in_ppo    = ppo_cols - random_cols
shared_cols    = random_cols & ppo_cols

print(f"  Random-only columns : {sorted(only_in_random) if only_in_random else 'None'}")
print(f"  PPO-only columns    : {sorted(only_in_ppo) if only_in_ppo else 'None'}")
print(f"  Shared columns      : {sorted(shared_cols)}")

# Check dtypes on shared columns
dtype_mismatches = []
for col in shared_cols:
    rd = random_df[col].dtype
    pd_ = ppo_df[col].dtype
    if rd != pd_:
        dtype_mismatches.append((col, rd, pd_))

if dtype_mismatches:
    print("\n  dtype mismatches on shared columns:")
    for col, rd, pd_ in dtype_mismatches:
        print(f"    {col}: random={rd}, ppo={pd_}")
else:
    print("  All shared columns have matching dtypes. Schema consistent.")


# Basic Profile: Size, Shape, File Size 
# Row count, column count, and file size give the reader a quick sense of dataset scale before any deeper analysis.

print("\n[2] BASIC PROFILE (Size & Shape)")
print("-" * 40)

def file_size_kb(path):
    return os.path.getsize(path) / 1024

print(f"  {'Metric':<25} {'Random':>12} {'PPO':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12}")
print(f"  {'Rows':<25} {len(random_df):>12,} {len(ppo_df):>12,}")
print(f"  {'Columns':<25} {len(random_df.columns):>12} {len(ppo_df.columns):>12}")
print(f"  {'File size (KB)':<25} {file_size_kb(RANDOM_PATH):>12.1f} {file_size_kb(PPO_PATH):>12.1f}")


#  Missing Values
# We compute the percentage of missing (NaN) values per column for each dataset.
# In a clean RL trajectory log we expect 0% missing everywhere any non-zero
# value would indicate a collection bug and needs to be flagged for the report.

print("\n[3] MISSING VALUES (% per column)")
print("-" * 40)

random_missing = (random_df.isnull().mean() * 100).round(4)
ppo_missing    = (ppo_df.isnull().mean() * 100).round(4)

# Align on shared columns for comparison
missing_df = pd.DataFrame({
    "column"       : sorted(shared_cols),
    "random_miss%" : [random_missing.get(c, float("nan")) for c in sorted(shared_cols)],
    "ppo_miss%"    : [ppo_missing.get(c, float("nan")) for c in sorted(shared_cols)],
})

print(missing_df.to_string(index=False))

any_missing = missing_df[["random_miss%", "ppo_miss%"]].gt(0).any().any()
if not any_missing:
    print("\n  No missing values detected in either dataset.")


# Outlier Detection (IQR method)
# We use the Interquartile Range (IQR) method to flag outliers:
#   Lower fence = Q1 - 1.5 * IQR
#   Upper fence = Q3 + 1.5 * IQR
# Any value outside those fences is counted as an outlier.
# We report the count and percentage per column, per dataset.
# This is a "quick check" as the assignment specifies — we're not removing
# or correcting anything, just surfacing potential data quality issues.

print("\n[4] OUTLIER CHECK (IQR method, shared numeric columns)")
print("-" * 40)

numeric_shared = [c for c in sorted(shared_cols)
                  if pd.api.types.is_numeric_dtype(random_df[c])]

outlier_records = []

for col in numeric_shared:
    for label, df in [("random", random_df), ("ppo", ppo_df)]:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        pct = round(n_outliers / len(df) * 100, 3)
        outlier_records.append({
            "column"   : col,
            "dataset"  : label,
            "outliers" : n_outliers,
            "outlier%" : pct,
            "fence_lo" : round(lo, 4),
            "fence_hi" : round(hi, 4),
        })

outlier_df = pd.DataFrame(outlier_records)
print(outlier_df.to_string(index=False))


# Descriptive Statistics
# Min, max, mean, and std for each shared numeric column gives the reader a
# feel for the value ranges. This feeds directly into the data dictionary
# section of the report (range of values, units, etc.).

print("\n[5] DESCRIPTIVE STATISTICS (shared numeric columns)")
print("-" * 40)

stats_rows = []
for col in numeric_shared:
    stats_rows.append({
        "column"      : col,
        "rand_min"    : round(random_df[col].min(), 4),
        "rand_max"    : round(random_df[col].max(), 4),
        "rand_mean"   : round(random_df[col].mean(), 4),
        "rand_std"    : round(random_df[col].std(), 4),
        "ppo_min"     : round(ppo_df[col].min(), 4),
        "ppo_max"     : round(ppo_df[col].max(), 4),
        "ppo_mean"    : round(ppo_df[col].mean(), 4),
        "ppo_std"     : round(ppo_df[col].std(), 4),
    })

stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))


# Export to CSV
# We build one combined summary CSV that can be dropped straight into the
# report. It merges the missing-value percentages and outlier counts together
# into a single row per column per dataset.

print("\n[6] EXPORTING SUMMARY TO", OUTPUT_PATH)
print("-" * 40)

export_rows = []
for col in numeric_shared:
    for label, df, miss_series in [
        ("random", random_df, random_missing),
        ("ppo",    ppo_df,    ppo_missing),
    ]:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()

        export_rows.append({
            "dataset"     : label,
            "column"      : col,
            "rows"        : len(df),
            "missing_%"   : miss_series.get(col, 0),
            "min"         : round(df[col].min(), 4),
            "max"         : round(df[col].max(), 4),
            "mean"        : round(df[col].mean(), 4),
            "std"         : round(df[col].std(), 4),
            "outliers"    : n_out,
            "outlier_%"   : round(n_out / len(df) * 100, 3),
        })

export_df = pd.DataFrame(export_rows)
export_df.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved {len(export_df)} rows to {OUTPUT_PATH}")
print("\nDone.")
