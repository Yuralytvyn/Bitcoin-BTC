import pandas as pd
import numpy as np
from ta import add_all_ta_features
from tsflex.features import FeatureCollection, FeatureDescriptor

# ===== CONFIG =====
TARGET_HORIZON = 3
MAX_LAGS = 50
ROLL_WINDOWS = [3, 5, 10, 20, 50]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
THRESHOLD = 0.005
SMOOTH_WINDOW = 3


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df.drop(["hour", "weekday"], axis=1, inplace=True)

    rng = df["high"] - df["low"]
    df["pos_in_range"] = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    df["pos_in_range"].fillna(0.5, inplace=True)

    df["intrabar_return"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["intrabar_return"].fillna(0, inplace=True)

    df["ma_5"] = df["close"].rolling(5, min_periods=1).mean().shift(1)
    df["ma_10"] = df["close"].rolling(10, min_periods=1).mean().shift(1)
    df["ma_50"] = df["close"].rolling(50, min_periods=1).mean().shift(1)

    df["ma_ratio_5_50"] = df["ma_5"] / df["ma_50"].replace(0, np.nan)
    df["ma_ratio_10_50"] = df["ma_10"] / df["ma_50"].replace(0, np.nan)

    df[["ma_ratio_5_50", "ma_ratio_10_50"]] = (
        df[["ma_ratio_5_50", "ma_ratio_10_50"]]
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    df["vol_ma_20"] = df["volume"].rolling(20, min_periods=1).mean().shift(1)
    df["volume_ratio"] = df["volume"] / df["vol_ma_20"].replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], 1.0).fillna(1.0)

    return df.drop(columns=["ma_5", "ma_10", "ma_50", "vol_ma_20"])


def generate_target(df: pd.DataFrame):
    smooth = (
        df["close"]
        .rolling(window=SMOOTH_WINDOW, min_periods=1)
        .mean()
        .shift(1)
    )

    future_smooth = smooth.shift(-TARGET_HORIZON)

    future_ret = future_smooth / smooth - 1

    target = pd.cut(
        future_ret,
        bins=[-np.inf, -THRESHOLD, THRESHOLD, np.inf],
        labels=[0, 1, 2],
    )

    return target

def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    for lag in range(1, MAX_LAGS + 1):
        df[f"lag_{lag}"] = df["close"].shift(lag)

    for w in ROLL_WINDOWS:
        df[f"rmean_{w}"] = df["close"].rolling(w, min_periods=1).mean().shift(1)
        df[f"rstd_{w}"] = df["close"].rolling(w, min_periods=1).std(ddof=0).shift(1)
        df[f"rmin_{w}"] = df["close"].rolling(w, min_periods=1).min().shift(1)
        df[f"rmax_{w}"] = df["close"].rolling(w, min_periods=1).max().shift(1)
        df[f"vol_{w}"] = df["log_return"].rolling(w, min_periods=1).std(ddof=0).shift(1)

        for q in QUANTILES:
            df[f"rq_{w}_{q}"] = df["close"].rolling(w, min_periods=1).quantile(q).shift(1)

    return df


def add_tsflex_features(df: pd.DataFrame) -> pd.DataFrame:
    def skew_func(x): return pd.Series(x).skew()
    def kurt_func(x): return pd.Series(x).kurt()
    def autocorr_func(x): return pd.Series(x).autocorr()

    fc = FeatureCollection([
        FeatureDescriptor(np.mean, "close", window=20, stride=1),
        FeatureDescriptor(np.std, "close", window=20, stride=1),
        FeatureDescriptor(np.median, "close", window=20, stride=1),
        FeatureDescriptor(np.max, "close", window=20, stride=1),
        FeatureDescriptor(np.min, "close", window=20, stride=1),
        FeatureDescriptor(skew_func, "close", window=20, stride=1),
        FeatureDescriptor(kurt_func, "close", window=20, stride=1),
        FeatureDescriptor(autocorr_func, "close", window=20, stride=1),
    ])

    res = fc.calculate(df, return_df=True).shift(1)
    res.columns = [f"tsf_{c}" for c in res.columns]

    return pd.concat([df, res], axis=1)


def daily_hourly_minute(file_list):
    for file_path in file_list:
        print(f"\n⚙️  Processing: {file_path}")
        df = pd.read_csv(file_path)
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["target"] = generate_target(df)

        df_base = df.copy()

        df_base = add_manual_features(df_base)
        df_base["log_return"].fillna(0, inplace=True)

        for lag in range(1, MAX_LAGS + 1):
            df_base[f"lag_{lag}"].fillna(df_base["close"].iloc[0], inplace=True)

        for w in ROLL_WINDOWS:
            df_base[f"rstd_{w}"].fillna(0, inplace=True)
            df_base[f"rmean_{w}"].fillna(df_base["close"], inplace=True)
            df_base[f"rmin_{w}"].fillna(df_base["close"], inplace=True)
            df_base[f"rmax_{w}"].fillna(df_base["close"], inplace=True)
            df_base[f"vol_{w}"].fillna(0, inplace=True)

            for q in QUANTILES:
                df_base[f"rq_{w}_{q}"].fillna(df_base["close"], inplace=True)

        df_base = add_advanced_features(df_base)

        df_base = add_all_ta_features(
            df_base,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            fillna=False,
        )

        # 🔴 FIX: remove ichimoku
        df_base = df_base.drop(
            columns=[col for col in df_base.columns if "ichimoku" in col.lower()],
            errors="ignore"
        )

        ta_columns = [
            col for col in df_base.columns
            if col.startswith(("volume_", "volatility_", "trend_", "momentum_", "others_"))
        ]

        for col in ta_columns:
            if df_base[col].isna().sum() == 0:
                continue

            if col.startswith(("volume_", "others_")):
                df_base[col].fillna(0, inplace=True)

            elif col.startswith("momentum_"):
                if "rsi" in col or "stoch" in col or "uo" in col:
                    df_base[col].fillna(50, inplace=True)
                else:
                    df_base[col].fillna(0, inplace=True)

            elif col.startswith("trend_"):
                df_base[col] = df_base[col].ffill().fillna(0)

            elif col.startswith("volatility_"):
                df_base[col].fillna(0, inplace=True)

        df_base = add_tsflex_features(df_base)

        tsf_columns = [col for col in df_base.columns if col.startswith("tsf_")]

        for col in tsf_columns:
            if df_base[col].isna().sum() == 0:
                continue

            if any(key in col for key in ["mean", "median", "min", "max"]):
                df_base[col] = df_base[col].ffill().fillna(df_base[col].iloc[0])
            else:
                df_base[col].fillna(0, inplace=True)

        df_base = df_base.dropna(subset=["target"])
        df_base = df_base.fillna(0)

        df_final = df_base.reset_index(drop=True)

        output_path = file_path.replace(".csv", "_feature.csv")
        df_final.to_csv(output_path, index=False)

        print(f"\n🎉 CLEAN DATASET READY → {df_final.shape}")
        print(f"📁 Saved to {output_path}\n")