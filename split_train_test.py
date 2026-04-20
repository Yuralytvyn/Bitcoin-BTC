import os
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from init import DATA_DIR, VALIDATION, VALIDATION_SIZE, PREDICTION_WINDOW

# =============================================================================
# Configuration
# =============================================================================
FILENAME: str = "btc_hourly_5y_feature.csv"
TEST_SIZE: float = 0.2

# Use the configured prediction window when available.
# This is the gap between train/val and test to avoid target contamination
# near the split boundary.
if isinstance(PREDICTION_WINDOW, (int, float)):
    GAP_STEPS: int = int(PREDICTION_WINDOW)
else:
    GAP_STEPS: int = 3


# =============================================================================
# Utils
# =============================================================================

def _print_class_distribution(name: str, df: pd.DataFrame) -> None:
    print(f"\n📊 CLASS DISTRIBUTION — {name.upper()}:")

    if "target" not in df.columns:
        print("❌ No 'target' column found!")
        return

    vc = df["target"].value_counts(dropna=False).sort_index()
    print(vc.to_string())

    for cls in [0, 1, 2]:
        if cls not in vc.index:
            print(f"⚠️ WARNING: Class {cls} is MISSING from {name} set!")


def _print_normalized_distribution(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
) -> None:
    print("\n🔍 NORMALIZED DISTRIBUTION:")

    print("\nTRAIN:")
    print(train_df["target"].value_counts(normalize=True))

    if val_df is not None:
        print("\nVAL:")
        print(val_df["target"].value_counts(normalize=True))

    print("\nTEST:")
    print(test_df["target"].value_counts(normalize=True))


def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df.reset_index(drop=True)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


# =============================================================================
# Scaling
# =============================================================================

def apply_scaling(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    scaler = StandardScaler()

    exclude_cols = {"target", "timestamp", "open_time", "close_time"}
    price_cols = {"open", "high", "low", "close", "volume"}

    feature_cols = [
        col for col in train_df.columns
        if col not in exclude_cols
        and col not in price_cols
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    print(f"\n🧠 Scaling {len(feature_cols)} features")

    scaler.fit(train_df[feature_cols])

    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()

    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    if val_df is not None:
        val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])

    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    return train_df, val_df, test_df


# =============================================================================
# Split
# =============================================================================

def train_test_split(
    filename: str = FILENAME,
    test_size: float = TEST_SIZE,
    create_validation: bool = VALIDATION,
    validation_size: float = VALIDATION_SIZE,
    gap_steps: int = GAP_STEPS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Chronological split for time-series data.

    Important:
    - no shuffling
    - train/val are strictly earlier than test
    - gap_steps removes boundary rows so labels near the split
      cannot use future values from the next segment
    """
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df = _ensure_time_sorted(df)

    n = len(df)
    test_split_idx = int(n * (1.0 - test_size))

    test_start = test_split_idx
    gap_start = max(0, test_start - gap_steps)

    if not create_validation:
        train_df = df.iloc[:gap_start].copy()
        test_df = df.iloc[test_start:].copy()
        val_df = None

    else:
        pre_test_end = gap_start

        val_len = int(pre_test_end * validation_size)
        val_end = pre_test_end
        val_start = max(0, val_end - val_len)

        train_end = max(0, val_start - gap_steps)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.reset_index(drop=True)

    print("\n🧱 SPLIT INFO")
    print(f"train: {len(train_df)}")
    print(f"val  : {0 if val_df is None else len(val_df)}")
    print(f"test : {len(test_df)}")

    _print_class_distribution("TRAIN", train_df)
    if val_df is not None:
        _print_class_distribution("VAL", val_df)
    _print_class_distribution("TEST", test_df)

    _print_normalized_distribution(train_df, val_df, test_df)

    print("\n-------------------------------------------------------\n")

    train_df, val_df, test_df = apply_scaling(train_df, val_df, test_df)

    base_name = filename.rsplit(".csv", 1)[0]

    train_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_test.csv"), index=False)

    if val_df is not None:
        val_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_val.csv"), index=False)

    return train_df, test_df, val_df