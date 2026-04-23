"""
run_preprocessing.py
====================
How to run numcompute_demo_data.csv through preprocessing.py

USAGE (from your terminal):
    python run_preprocessing.py

Make sure preprocessing.py is in the SAME folder as this file,
or adjust the import path at the top.

Steps this script covers:
  1. Read the CSV (pure Python + NumPy, no pandas)
  2. SimpleImputer  — fill the 4 missing numeric values
  3. StandardScaler — z-score scale the numeric features
  4. MinMaxScaler   — scale to [0, 1]
  5. OneHotEncoder  — encode 'city' and 'study_mode' columns
"""

import numpy as np

# ─── IMPORT YOUR preprocessing.py ────────────────────────────────────────────
# If preprocessing.py is in the same folder as this script:
from preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

# If it lives inside a package folder called 'numcompute', use this instead:
# from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
# ─────────────────────────────────────────────────────────────────────────────


CSV_PATH = "numcompute_demo_data.csv"   # <-- put your CSV in the same folder


# ===========================================================================
# STEP 1 — Read the CSV manually (no pandas needed)
# ===========================================================================

print("\n" + "=" * 60)
print("STEP 1 — Reading the CSV file")
print("=" * 60)

with open(CSV_PATH, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# First line is the header
headers = [h.strip() for h in lines[0].split(",")]
print(f"Columns found: {headers}")

# ── Separate numeric columns from text columns ────────────────────────────
# Numeric columns we care about for ML (by index in the CSV):
#   0=id, 1=age, 2=study_hours, 3=attendance_pct, 4=previous_score,
#   7=assignment_score, 8=exam_score, 9=final_score,
#   10=passed_true, 11=passed_pred, 12=pass_prob
# Text columns:
#   5=city, 6=study_mode

NUMERIC_COL_INDICES = [1, 2, 3, 4, 7, 8, 9]     # features we will scale
NUMERIC_COL_NAMES   = ["age", "study_hours", "attendance_pct",
                        "previous_score", "assignment_score",
                        "exam_score", "final_score"]

raw_numeric = []   # will hold float rows (NaN where data is missing)
city_list   = []   # raw city strings per row
mode_list   = []   # raw study_mode strings per row

for line in lines[1:]:                    # skip header
    cells = [c.strip() for c in line.split(",")]

    # Parse numeric columns — empty string → NaN
    num_row = []
    for idx in NUMERIC_COL_INDICES:
        cell = cells[idx] if idx < len(cells) else ""
        try:
            num_row.append(float(cell))
        except ValueError:
            num_row.append(np.nan)        # missing value
    raw_numeric.append(num_row)

    # Save text columns as-is
    city_list.append(cells[5] if len(cells) > 5 else "")
    mode_list.append(cells[6] if len(cells) > 6 else "")

# Convert to NumPy array
X_raw = np.array(raw_numeric, dtype=float)

print(f"\nLoaded shape : {X_raw.shape}  (rows × numeric features)")
print(f"Missing (NaN): {int(np.isnan(X_raw).sum())} cells")

# Show which columns have missing values
for j, name in enumerate(NUMERIC_COL_NAMES):
    n_missing = int(np.isnan(X_raw[:, j]).sum())
    if n_missing:
        print(f"  → '{name}' has {n_missing} missing value(s)")


# ===========================================================================
# STEP 2 — SimpleImputer : fill NaNs before scaling
# ===========================================================================

print("\n" + "=" * 60)
print("STEP 2 — SimpleImputer  (strategy='mean')")
print("=" * 60)

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_raw)

print("Fill values learned per column:")
for name, val in zip(NUMERIC_COL_NAMES, imputer.statistics_):
    print(f"  {name:<22} → {val:.4f}")

print(f"\nNaN count after imputation: {int(np.isnan(X_imputed).sum())}")


# ===========================================================================
# STEP 3 — StandardScaler : zero mean, unit variance
# ===========================================================================

print("\n" + "=" * 60)
print("STEP 3 — StandardScaler  (z-score)")
print("=" * 60)

std_scaler = StandardScaler()
X_standard = std_scaler.fit_transform(X_imputed)

print(f"{'Column':<22} {'Mean (before)':>14} {'Std (before)':>13} "
      f"{'Mean (after)':>13} {'Std (after)':>12}")
print("-" * 76)

orig_mean = np.nanmean(X_raw, axis=0)
orig_std  = np.nanstd(X_raw,  axis=0)

for name, om, os_, sm, ss_ in zip(
    NUMERIC_COL_NAMES, orig_mean, orig_std,
    X_standard.mean(axis=0), X_standard.std(axis=0)
):
    print(f"  {name:<20} {om:>14.3f} {os_:>13.3f} {sm:>13.6f} {ss_:>12.6f}")

print(f"\nAll means ≈ 0 ? {np.allclose(X_standard.mean(axis=0), 0, atol=1e-10)}")
print(f"All stds  ≈ 1 ? {np.allclose(X_standard.std(axis=0),  1, atol=1e-10)}")

# inverse_transform check
X_back = std_scaler.inverse_transform(X_standard)
print(f"Inverse recovers original? {np.allclose(X_back, X_imputed)}")


# ===========================================================================
# STEP 4 — MinMaxScaler : compress everything to [0, 1]
# ===========================================================================

print("\n" + "=" * 60)
print("STEP 4 — MinMaxScaler  (feature_range=[0, 1])")
print("=" * 60)

mm_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_minmax  = mm_scaler.fit_transform(X_imputed)

print(f"{'Column':<22} {'Data Min':>10} {'Data Max':>10} "
      f"{'Scaled Min':>11} {'Scaled Max':>11}")
print("-" * 68)

for name, dmin, dmax, smin, smax in zip(
    NUMERIC_COL_NAMES,
    mm_scaler.data_min_, mm_scaler.data_max_,
    X_minmax.min(axis=0), X_minmax.max(axis=0)
):
    print(f"  {name:<20} {dmin:>10.2f} {dmax:>10.2f} {smin:>11.6f} {smax:>11.6f}")

print(f"\nAll mins = 0? {np.allclose(X_minmax.min(axis=0), 0.0)}")
print(f"All maxs = 1? {np.allclose(X_minmax.max(axis=0), 1.0)}")

# inverse_transform check
print(f"Inverse recovers original? "
      f"{np.allclose(mm_scaler.inverse_transform(X_minmax), X_imputed)}")


# ===========================================================================
# STEP 5 — OneHotEncoder : encode 'city' and 'study_mode'
# ===========================================================================

print("\n" + "=" * 60)
print("STEP 5 — OneHotEncoder  (city  +  study_mode)")
print("=" * 60)

# ── Convert string categories to integer codes ────────────────────────────
# OneHotEncoder works on numbers, so we map strings → integers first.

def strings_to_int_array(str_list):
    """
    Turn a list of strings into a (n, 1) integer NumPy array.
    Empty / missing strings → NaN so the imputer can fill them.
    Returns (int_array, label_list) where label_list[i] is the string
    for integer code i.
    """
    unique_labels = sorted(set(s for s in str_list if s))  # sorted unique non-empty
    label_to_int  = {label: i for i, label in enumerate(unique_labels)}
    int_col = []
    for s in str_list:
        if s:
            int_col.append(float(label_to_int[s]))
        else:
            int_col.append(np.nan)                # missing → NaN
    return np.array(int_col, dtype=float).reshape(-1, 1), unique_labels


city_int, city_labels = strings_to_int_array(city_list)
mode_int, mode_labels = strings_to_int_array(mode_list)

print(f"Unique cities     : {city_labels}")
print(f"Unique study modes: {mode_labels}")

# Fill any missing city / mode with the most common value
city_int = SimpleImputer(strategy="most_frequent").fit_transform(city_int)
mode_int = SimpleImputer(strategy="most_frequent").fit_transform(mode_int)

# One-hot encode
enc_city = OneHotEncoder(drop_first=False)
enc_mode = OneHotEncoder(drop_first=False)

X_city = enc_city.fit_transform(city_int)
X_mode = enc_mode.fit_transform(mode_int)

city_col_names = [f"city_{c}"  for c in city_labels]
mode_col_names = [f"mode_{m}"  for m in mode_labels]

print(f"\nCity OHE matrix shape : {X_city.shape}  → columns: {city_col_names}")
print(f"Mode OHE matrix shape : {X_mode.shape}  → columns: {mode_col_names}")

# Show first 5 rows as a neat table
print(f"\nFirst 5 rows — city one-hot encoding:")
header_line = f"  {'original city':<16}" + "".join(f"{c:<16}" for c in city_col_names)
print(header_line)
print("  " + "-" * (len(header_line) - 2))
for i in range(5):
    row_str = f"  {city_list[i]:<16}" + "".join(f"{int(v):<16}" for v in X_city[i])
    print(row_str)


# ===========================================================================
# STEP 6 — Assemble the final feature matrix
# ===========================================================================

print("\n" + "=" * 60)
print("STEP 6 — Final feature matrix  (StandardScaled + OHE)")
print("=" * 60)

# Combine z-score scaled numerics with the one-hot columns
X_final      = np.hstack([X_standard, X_city, X_mode])
final_columns = NUMERIC_COL_NAMES + city_col_names + mode_col_names

print(f"Shape : {X_final.shape}")
print(f"  {len(NUMERIC_COL_NAMES)} numeric (z-scored)  +  "
      f"{len(city_col_names)} city OHE  +  "
      f"{len(mode_col_names)} mode OHE")

print(f"\nAll {len(final_columns)} columns:")
for i, col in enumerate(final_columns):
    print(f"  [{i:02d}] {col}")

print(f"\nFirst 3 rows sample:")
col_w = 16
print("  " + "".join(f"{c[:col_w-1]:<{col_w}}" for c in final_columns))
for row in X_final[:3]:
    print("  " + "".join(f"{v:<{col_w}.4f}" for v in row))

print("\n" + "=" * 60)
print("Done! All preprocessing steps completed successfully ✓")
print("=" * 60) 