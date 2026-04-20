import os
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from init import DATA_DIR, VALIDATION, VALIDATION_SIZE, PREDICTION_WINDOW

FILENAME: str = "btc_hourly_5y_feature.csv"
TEST_SIZE: float = 0.2

# prediction horizon (на скільки вперед дивиться target)
HORIZON = int(PREDICTION_WINDOW) if isinstance(PREDICTION_WINDOW, (int, float)) else 1


# =============================================================================
# UTILS
# =============================================================================

def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df.reset_index(drop=True)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def _drop_leakage_tail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляємо останні HORIZON рядків,
    бо їх target дивиться в майбутнє (якого немає в датасеті)
    """
    if HORIZON <= 0:
        return df
    return df.iloc[:-HORIZON].copy()


# =============================================================================
# SCALING (БЕЗ LEAKAGE)
# =============================================================================

def apply_scaling(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:

    scaler = StandardScaler()

    exclude_cols = {"target", "timestamp", "open_time", "close_time"}

    feature_cols = [
        col for col in train_df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    print(f"\n🧠 Scaling {len(feature_cols)} features")

    # FIT ONLY ON TRAIN
    scaler.fit(train_df[feature_cols])

    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    if val_df is not None:
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))

    return train_df, val_df, test_df


# =============================================================================
# SPLIT (БЕЗ LEAKAGE)
# =============================================================================

def train_test_split(
    filename: str = FILENAME,
    test_size: float = TEST_SIZE,
    create_validation: bool = VALIDATION,
    validation_size: float = VALIDATION_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    df = _ensure_time_sorted(df)

    # ❗ КРИТИЧНО: прибираємо хвіст де target дивиться в майбутнє
    df = _drop_leakage_tail(df)

    df = df.dropna(subset=["target"]).reset_index(drop=True)

    n = len(df)

    # ❗ РЕАЛЬНИЙ SAFE SPLIT
    test_start = int(n * (1.0 - test_size))

    # ❗ головне виправлення:
    # ми відсуваємо train назад на HORIZON
    train_end = test_start - HORIZON

    if train_end <= 0:
        raise ValueError("Not enough data after applying horizon gap")

    if not create_validation:
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:].copy()
        val_df = None

    else:
        val_size_abs = int(train_end * validation_size)

        val_end = train_end
        val_start = val_end - val_size_abs

        train_df = df.iloc[:val_start].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

    print("\n🧱 SAFE SPLIT")
    print(f"train: {len(train_df)}")
    print(f"val  : {0 if val_df is None else len(val_df)}")
    print(f"test : {len(test_df)}")

    # =========================
    # SCALING
    # =========================
    train_df, val_df, test_df = apply_scaling(train_df, val_df, test_df)

    # =========================
    # SAVE
    # =========================
    base_name = filename.rsplit(".csv", 1)[0]

    train_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_test.csv"), index=False)

    if val_df is not None:
        val_df.to_csv(os.path.join(DATA_DIR, f"{base_name}_val.csv"), index=False)

    return train_df, test_df, val_df