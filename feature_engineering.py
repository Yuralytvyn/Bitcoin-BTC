import pandas as pd
import numpy as np
from pathlib import Path
from ta import add_all_ta_features
from tsflex.features import FeatureCollection, FeatureDescriptor

# ===== CONFIG =====
TARGET_HORIZON = 3
MAX_LAGS = 50
ROLL_WINDOWS = [3, 5, 10, 20, 50]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
THRESHOLD = 0.005
SMOOTH_WINDOW = 3
TSF_WINDOW = 20


def _ensure_sorted_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must contain a 'timestamp' column.")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def generate_target(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    threshold: float = THRESHOLD,
    smooth_window: int = SMOOTH_WINDOW,
) -> pd.Series:
    """
    Target uses only past data at time t and future data at time t + horizon.
    Future values are used only for labeling, never as features.
    """
    close = df["close"].astype(float)

    # Past-only smoothing.
    smooth = close.rolling(window=smooth_window, min_periods=1).mean().shift(1)

    # Future value used only for label creation.
    future_smooth = smooth.shift(-horizon)
    future_ret = (future_smooth / smooth) - 1

    target = pd.cut(
        future_ret,
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 1, 2],
    )

    return target.astype("float")


def add_manual_features(df: pd.DataFrame, max_lags: int = MAX_LAGS) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)

    out["log_return"] = np.log(close / close.shift(1).replace(0, np.nan))

    for lag in range(1, max_lags + 1):
        out[f"lag_{lag}"] = close.shift(lag)

    for w in ROLL_WINDOWS:
        out[f"rmean_{w}"] = close.rolling(w, min_periods=1).mean().shift(1)
        out[f"rstd_{w}"] = close.rolling(w, min_periods=1).std(ddof=0).shift(1)
        out[f"rmin_{w}"] = close.rolling(w, min_periods=1).min().shift(1)
        out[f"rmax_{w}"] = close.rolling(w, min_periods=1).max().shift(1)
        out[f"vol_{w}"] = out["log_return"].rolling(w, min_periods=1).std(ddof=0).shift(1)

        for q in QUANTILES:
            out[f"rq_{w}_{q}"] = close.rolling(w, min_periods=1).quantile(q).shift(1)

    return out


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    out["hour"] = out["timestamp"].dt.hour
    out["weekday"] = out["timestamp"].dt.weekday
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out = out.drop(columns=["hour", "weekday"])

    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)

    rng = high - low
    out["pos_in_range"] = (close - low) / rng.replace(0, np.nan)
    out["pos_in_range"] = out["pos_in_range"].fillna(0.5)

    out["intrabar_return"] = (close - open_) / open_.replace(0, np.nan)
    out["intrabar_return"] = out["intrabar_return"].fillna(0.0)

    out["ma_5"] = close.rolling(5, min_periods=1).mean().shift(1)
    out["ma_10"] = close.rolling(10, min_periods=1).mean().shift(1)
    out["ma_50"] = close.rolling(50, min_periods=1).mean().shift(1)

    out["ma_ratio_5_50"] = out["ma_5"] / out["ma_50"].replace(0, np.nan)
    out["ma_ratio_10_50"] = out["ma_10"] / out["ma_50"].replace(0, np.nan)
    out[["ma_ratio_5_50", "ma_ratio_10_50"]] = (
        out[["ma_ratio_5_50", "ma_ratio_10_50"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    out["vol_ma_20"] = volume.rolling(20, min_periods=1).mean().shift(1)
    out["volume_ratio"] = volume / out["vol_ma_20"].replace(0, np.nan)
    out["volume_ratio"] = out["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return out.drop(columns=["ma_5", "ma_10", "ma_50", "vol_ma_20"])


def add_tsflex_features(df: pd.DataFrame) -> pd.DataFrame:
    def skew_func(x):
        return pd.Series(x).skew()

    def kurt_func(x):
        return pd.Series(x).kurt()

    def autocorr_func(x):
        return pd.Series(x).autocorr()

    fc = FeatureCollection(
        [
            FeatureDescriptor(np.mean, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(np.std, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(np.median, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(np.max, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(np.min, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(skew_func, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(kurt_func, "close", window=TSF_WINDOW, stride=1),
            FeatureDescriptor(autocorr_func, "close", window=TSF_WINDOW, stride=1),
        ]
    )

    feature_input = df[["close"]].copy()
    res = fc.calculate(feature_input, return_df=True).shift(1)
    res.columns = [f"tsf_{c}" for c in res.columns]

    return pd.concat([df, res], axis=1)


def _fill_leading_nans_causally(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "log_return" in out.columns:
        out["log_return"] = out["log_return"].fillna(0.0)

    for lag in range(1, MAX_LAGS + 1):
        col = f"lag_{lag}"
        if col in out.columns:
            out[col] = out[col].fillna(out["close"].iloc[0])

    for w in ROLL_WINDOWS:
        for col in [f"rstd_{w}", f"vol_{w}"]:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)

        for col in [f"rmean_{w}", f"rmin_{w}", f"rmax_{w}"]:
            if col in out.columns:
                out[col] = out[col].ffill().fillna(out["close"].iloc[0])

        for q in QUANTILES:
            col = f"rq_{w}_{q}"
            if col in out.columns:
                out[col] = out[col].ffill().fillna(out["close"].iloc[0])

    ta_columns = [
        col
        for col in out.columns
        if col.startswith(("volume_", "volatility_", "trend_", "momentum_", "others_"))
    ]

    for col in ta_columns:
        if not out[col].isna().any():
            continue

        if col.startswith(("volume_", "others_")):
            out[col] = out[col].fillna(0.0)

        elif col.startswith("momentum_"):
            if any(key in col.lower() for key in ("rsi", "stoch", "uo")):
                out[col] = out[col].fillna(50.0)
            else:
                out[col] = out[col].fillna(0.0)

        elif col.startswith("trend_"):
            out[col] = out[col].ffill().fillna(0.0)

        elif col.startswith("volatility_"):
            out[col] = out[col].fillna(0.0)

    tsf_columns = [col for col in out.columns if col.startswith("tsf_")]
    for col in tsf_columns:
        if not out[col].isna().any():
            continue

        if any(key in col.lower() for key in ("mean", "median", "min", "max")):
            out[col] = out[col].ffill().fillna(out["close"].iloc[0])
        else:
            out[col] = out[col].fillna(0.0)

    return out


def process_file(file_path: str, target_horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    print(f"\n⚙️ Processing: {file_path}")

    df = pd.read_csv(file_path)
    df = _ensure_sorted_datetime(df)

    # Label only; never used as a feature.
    df["target"] = generate_target(df, horizon=target_horizon)

    df = add_manual_features(df)
    df = add_advanced_features(df)

    df = add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=False,
    )

    # Remove future-looking Ichimoku outputs.
    df = df.drop(columns=[c for c in df.columns if "ichimoku" in c.lower()], errors="ignore")

    df = add_tsflex_features(df)

    # Fill only with values that do not use the future.
    df = _fill_leading_nans_causally(df)

    # Drop rows that cannot be labeled because the future horizon is missing.
    df = df.dropna(subset=["target"])

    # Clean residual NaNs after causal fills.
    df = df.fillna(0)

    df["target"] = df["target"].astype(int)

    df_final = df.reset_index(drop=True)

    output_path = str(Path(file_path).with_suffix("")) + "_feature.csv"
    df_final.to_csv(output_path, index=False)

    print(f"🎉 CLEAN DATASET READY → {df_final.shape}")
    print(f"📁 Saved to {output_path}")

    return df_final


def daily_hourly_minute(file_list):
    """
    Backward-compatible wrapper for your current pipeline.
    """
    results = []
    for file_path in file_list:
        results.append(process_file(file_path))
    return results