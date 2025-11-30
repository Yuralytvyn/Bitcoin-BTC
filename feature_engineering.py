"""
Extended feature engineering for cryptocurrency datasets.

This module provides an updated implementation of the `daily_hourly_minute` function
that reads raw OHLCV (Open‑High‑Low‑Close‑Volume) CSV files from `DATA_DIR`,
enriches them with a wide range of technical indicators and price/volume
statistics, generates a 3‑class target variable, and saves the resulting
feature matrix back to disk.  It builds upon an existing pipeline by adding
several additional indicators commonly used in quantitative finance, such as
stochastic oscillators, average true range, MACD, on‑balance volume, candlestick
features, momentum ratios and moving‑average differences.
"""

import os
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from init import DATA_DIR  # type: ignore
except Exception:
    # Fall back to current working directory if DATA_DIR is not defined
    DATA_DIR = os.getcwd()


def _add_candle_features(df: pd.DataFrame) -> None:
    """Add candlestick pattern features to the DataFrame.

    This includes raw price range, body size, upper and lower wicks, and
    normalised versions of body and wicks (scaled by price range).
    """
    df['price_range'] = df['high'] - df['low']
    df['body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    # Avoid division by zero: replace zero ranges with NaN to prevent inf
    pr = df['price_range'].replace(0, np.nan)
    df['body_norm'] = df['body'] / pr
    df['upper_wick_norm'] = df['upper_wick'] / pr
    df['lower_wick_norm'] = df['lower_wick'] / pr


def _add_stochastic(df: pd.DataFrame, window: int = 14) -> None:
    """Add stochastic oscillator %K and %D columns.

    `window` defines the lookback period used to compute the highest high and
    lowest low.  A 3‑period moving average of %K yields %D.
    """
    high_roll = df['high'].rolling(window).max()
    low_roll = df['low'].rolling(window).min()
    stoch_k = 100 * (df['close'] - low_roll) / (high_roll - low_roll)
    df[f'stoch_k_{window}'] = stoch_k
    df[f'stoch_d_{window}'] = stoch_k.rolling(3).mean()


def _add_atr(df: pd.DataFrame, window: int = 14) -> None:
    """Add Average True Range (ATR) over the specified window."""
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f'atr_{window}'] = tr.rolling(window).mean()


def _add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
    """Add MACD and signal line columns."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    df[f'macd_{fast}_{slow}'] = macd
    df[f'macd_signal_{fast}_{slow}_{signal}'] = macd.ewm(span=signal, adjust=False).mean()


def _add_obv(df: pd.DataFrame) -> None:
    """Add On‑Balance Volume (OBV) column."""
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i - 1, 'close']:
            obv[i] = obv[i - 1] + df.loc[i, 'volume']
        elif df.loc[i, 'close'] < df.loc[i - 1, 'close']:
            obv[i] = obv[i - 1] - df.loc[i, 'volume']
        else:
            obv[i] = obv[i - 1]
    df['obv'] = obv


def _add_typical_prices(df: pd.DataFrame) -> None:
    """Add typical and weighted price columns."""
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['weighted_price'] = (df['high'] + df['low'] + 2.0 * df['close']) / 4.0


def _add_volume_ratios(df: pd.DataFrame, windows: Iterable[int]) -> None:
    """Add volume ratio and difference relative to rolling means."""
    for w in windows:
        ma_col = f'vol_ma_{w}'
        if ma_col not in df.columns:
            df[ma_col] = df['volume'].rolling(w).mean()
        df[f'vol_ratio_{w}'] = df['volume'] / df[ma_col]
        df[f'vol_diff_{w}'] = df['volume'] - df[ma_col]


def _add_momentum_ratios(df: pd.DataFrame, periods: Iterable[int]) -> None:
    """Add percentage change over multiple lookback periods."""
    for p in periods:
        df[f'momentum_ratio_{p}'] = df['close'].pct_change(p)


def _add_ma_differences(df: pd.DataFrame, short_windows: Iterable[int], long_windows: Iterable[int]) -> None:
    """Add differences between short and long rolling means."""
    for s in short_windows:
        for l in long_windows:
            if s < l:
                short_col = f'rolling_mean_{s}'
                long_col = f'rolling_mean_{l}'
                # Ensure both moving averages exist
                if short_col in df.columns and long_col in df.columns:
                    df[f'ma_diff_{s}_{l}'] = df[short_col] - df[long_col]


def daily_hourly_hourly() -> None:
    """Process each CSV file in DATA_DIR and generate enhanced features.

    The function reads each file, sorts it by timestamp, computes a variety of
    time‑series and technical indicators (including the original features and
    newly added ones), and writes the enriched dataset back to a CSV file
    with a `_feature.csv` suffix.
    """
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        # Only process CSV files
        if not filename.lower().endswith('.csv'):
            continue
        print(f"\nProcessing: {filename}")
        df = pd.read_csv(path)

        # -----------------------------------------------------------
        # TIME CONVERSION
        # -----------------------------------------------------------
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])

        fname = filename.lower()

        # =======================================================
        # TLAG — time difference feature
        # =======================================================
        df = df.sort_values("timestamp")
        df["tlag_sec"] = df["timestamp"].diff().dt.total_seconds()
        if "daily" in fname:
            df["tlag"] = df["timestamp"].diff().dt.days
        elif "hour" in fname:
            df["tlag"] = df["timestamp"].diff().dt.total_seconds() / 3600
        elif "min" in fname or "minute" in fname:
            df["tlag"] = df["timestamp"].diff().dt.total_seconds() / 60
        else:
            df["tlag"] = df["tlag_sec"]

        # =======================================================
        # TIMEFRAME‑DEPENDENT FEATURES
        # =======================================================
        if "daily" in fname:
            df['dayofyear'] = df['timestamp'].dt.dayofyear
            df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
            df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
            df['weekday'] = df['timestamp'].dt.weekday
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        elif "hour" in fname:
            df['hour'] = df['timestamp'].dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['weekday'] = df['timestamp'].dt.weekday
            df['weekday_hour'] = df['weekday'] * 24 + df['hour']
            df['weekhour_sin'] = np.sin(2 * np.pi * df['weekday_hour'] / (7 * 24))
            df['weekhour_cos'] = np.cos(2 * np.pi * df['weekday_hour'] / (7 * 24))
        elif "min" in fname or "minute" in fname:
            df['minute'] = df['timestamp'].dt.minute
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
            df['second'] = df['timestamp'].dt.second

        # =======================================================
        # NUMERIC CASTING
        # =======================================================
        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote'
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)

        # =======================================================
        # BASIC PRICE FEATURES
        # =======================================================
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_abs'] = df['log_return'].abs()
        df['time_index'] = np.arange(len(df))
        for w in [3, 7, 14]:
            df[f'volatility_{w}'] = df['log_return'].rolling(w).std()

        # =======================================================
        # ROLLING MEANS & VOLATILITY
        # =======================================================
        for window in [5, 10, 20, 50, 100]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window).std()

        # =======================================================
        # VOLUME MOVING AVERAGES
        # =======================================================
        for w in [5, 10]:
            df[f'vol_ma_{w}'] = df['volume'].rolling(w).mean()

        # =======================================================
        # RSI
        # =======================================================
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # =======================================================
        # BOLLINGER BANDS
        # =======================================================
        df['bb_mid'] = df['rolling_mean_20']
        df['bb_std'] = df['rolling_std_20']
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # =======================================================
        # LAG FEATURES
        # =======================================================
        for lag in range(1, 4):
            df[f'close_lag{lag}'] = df['close'].shift(lag)
        for lag in range(1, 3):
            df[f'volume_lag{lag}'] = df['volume'].shift(lag)

        # =======================================================
        # EMA
        # =======================================================
        for span in [5, 10, 20, 50, 100]:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

        # =======================================================
        # MOMENTUM (difference)
        # =======================================================
        for p in [3, 5, 10, 20]:
            df[f'momentum_{p}'] = df['close'] - df['close'].shift(p)

        # =======================================================
        # VWAP
        # =======================================================
        df["vwap_20"] = (
            (df["close"] * df["volume"]).rolling(20).sum() /
            df["volume"].rolling(20).sum()
        )

        # =======================================================
        # PRICE VS MA
        # =======================================================
        for w in [5, 10, 20]:
            df[f'close_vs_ma_{w}'] = df['close'] / df[f'rolling_mean_{w}'] - 1

        # ------------------------------------------------------------------
        # ADDITIONAL FEATURES
        # ------------------------------------------------------------------
        # Candlestick features
        _add_candle_features(df)
        # Stochastic oscillator and ATR
        _add_stochastic(df, window=14)
        _add_atr(df, window=14)
        # MACD and signal
        _add_macd(df)
        # OBV
        _add_obv(df)
        # Typical and weighted prices
        _add_typical_prices(df)
        # Volume ratio and difference relative to moving averages (includes MA20)
        _add_volume_ratios(df, windows=[5, 10, 20])
        # Momentum ratios (percentage change)
        _add_momentum_ratios(df, periods=[3, 5, 10, 20])
        # Moving average differences
        _add_ma_differences(df, short_windows=[5, 10], long_windows=[20, 50, 100])

        # =======================================================
        # 3‑CLASS TARGET (target = -1, 0, 1)
        # =======================================================
        df["future_close"] = df["close"].shift(-1)
        df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
        threshold = 0.001  # 0.1%
        df["target"] = 0
        df.loc[df["future_return"] > threshold, "target"] = 1
        df.loc[df["future_return"] < -threshold, "target"] = -1
        df["target"] = df["target"] + 1

        # =======================================================
        # CLEANING
        # =======================================================
        df.drop(['timestamp', 'open_time', 'close_time'], axis=1, inplace=True)
        df.drop(columns=["future_close", "future_return"], inplace=True)
        # Forward fill then replace any remaining NaNs with zero
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # =======================================================
        # SAVE
        # =======================================================
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(DATA_DIR, base + "_feature.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


