import os
from typing import Optional, Tuple

import pandas as pd
from init import DATA_DIR, VALIDATION, VALIDATION_SIZE

# =============================================================================
# Configuration
# =============================================================================

FILENAME: str = "btc_hourly_5y_feature.csv"
TEST_SIZE: float = 0.2


def _print_class_distribution(name: str, df: pd.DataFrame):
    """Helper: print how many samples of each target class exist."""
    print(f"\n📊 CLASS DISTRIBUTION — {name.upper()}:")
    if "target" not in df.columns:
        print("❌ No 'target' column found!")
        return

    vc = df["target"].value_counts(dropna=False).sort_index()
    print(vc.to_string())

    # Check for missing classes
    for cls in [0, 1, 2]:
        if cls not in vc.index:
            print(f"⚠️ WARNING: Class {cls} is MISSING from {name} set!")


def train_test_split(
    filename: str = FILENAME,
    test_size: float = TEST_SIZE,
    create_validation: bool = VALIDATION,
    validation_size: float = VALIDATION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    # Load full dataset
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    # --- Drop rows with missing targets (VERY important for stability)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    n_samples = len(df)

    # Chronological split
    test_split_idx = int(n_samples * (1.0 - test_size))
    train_df = df.iloc[:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    val_df: Optional[pd.DataFrame] = None

    if create_validation:
        val_length = int(len(train_df) * validation_size)
        val_df = train_df.iloc[-val_length:].copy()
        train_df = train_df.iloc[:-val_length].copy()

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 🔥 CRITICAL DIAGNOSTICS — check 3-class coverage for each split
    # ---------------------------------------------------------------------
    _print_class_distribution("TRAIN", train_df)
    if val_df is not None:
        _print_class_distribution("VAL", val_df)
    _print_class_distribution("TEST", test_df)
    print("\n-------------------------------------------------------\n")

    # Save
    base_name = filename.rsplit(".csv", 1)[0]

    train_path = os.path.join(DATA_DIR, f"{base_name}_train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved: {train_path}")

    test_path = os.path.join(DATA_DIR, f"{base_name}_test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {test_path}")

    if val_df is not None:
        val_path = os.path.join(DATA_DIR, f"{base_name}_val.csv")
        val_df.to_csv(val_path, index=False)
        print(f"Saved: {val_path}")

    return train_df, test_df, val_df
