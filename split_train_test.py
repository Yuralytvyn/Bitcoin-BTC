import os
from typing import Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler
from init import DATA_DIR, VALIDATION, VALIDATION_SIZE

# =============================================================================
# Configuration
# =============================================================================

FILENAME: str = "btc_hourly_5y_feature.csv"
TEST_SIZE: float = 0.2


def _print_class_distribution(name: str, df: pd.DataFrame):
    print(f"\n\U0001F4CA CLASS DISTRIBUTION — {name.upper()}:")
    if "target" not in df.columns:
        print("❌ No 'target' column found!")
        return

    vc = df["target"].value_counts(dropna=False).sort_index()
    print(vc.to_string())

    for cls in [0, 1, 2]:
        if cls not in vc.index:
            print(f"⚠️ WARNING: Class {cls} is MISSING from {name} set!")


def apply_scaling(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], test_df: pd.DataFrame):
    scaler = StandardScaler()
    exclude_cols = {"target", "timestamp", "open_time", "close_time"}

    # Масштабуємо тільки числові колонки, які не в списку виключень
    feature_cols = [
        col for col in train_df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    if val_df is not None:
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    return train_df, val_df, test_df


def train_test_split(
    filename: str = FILENAME,
    test_size: float = TEST_SIZE,
    create_validation: bool = VALIDATION,
    validation_size: float = VALIDATION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    n_samples = len(df)
    test_split_idx = int(n_samples * (1.0 - test_size))
    train_df = df.iloc[:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    val_df: Optional[pd.DataFrame] = None
    if create_validation:
        val_length = int(len(train_df) * validation_size)
        val_df = train_df.iloc[-val_length:].copy()
        train_df = train_df.iloc[:-val_length].copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.reset_index(drop=True)

    _print_class_distribution("TRAIN", train_df)
    if val_df is not None:
        _print_class_distribution("VAL", val_df)
    _print_class_distribution("TEST", test_df)
    print("\n-------------------------------------------------------\n")

    # 🚀 Apply StandardScaler
    print("🧲 Scaling numerical features with StandardScaler...")
    train_df, val_df, test_df = apply_scaling(train_df, val_df, test_df)

    # Save
    base_name = filename.rsplit(".csv", 1)[0]
    train_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_test.csv"), index=False)
    if val_df is not None:
        val_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_val.csv"), index=False)

    return train_df, test_df, val_df
