import pandas as pd
import numpy as np
import os

from init import *
def daily_hourly_minute()->None:
    for filename in os.listdir(DATA_DIR):

        path = os.path.join(DATA_DIR, filename)
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
            df["tlag"] = df["timestamp"].diff().dt.total_seconds()

        # =======================================================
        # TIMEFRAME-DEPENDENT FEATURES
        # =======================================================

        if "daily" in fname:
            df['dayofyear'] = df['timestamp'].dt.dayofyear
            df['doy_sin'] = np.sin(2*np.pi*df['dayofyear']/365)
            df['doy_cos'] = np.cos(2*np.pi*df['dayofyear']/365)

            df['weekday'] = df['timestamp'].dt.weekday
            df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
            df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)

        elif "hour" in fname:
            df['hour'] = df['timestamp'].dt.hour
            df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
            df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

            df['weekday'] = df['timestamp'].dt.weekday
            df['weekday_hour'] = df['weekday']*24 + df['hour']
            df['weekhour_sin'] = np.sin(2*np.pi*df['weekday_hour']/(7*24))
            df['weekhour_cos'] = np.cos(2*np.pi*df['weekday_hour']/(7*24))

        elif "min" in fname or "minute" in fname:
            df['minute'] = df['timestamp'].dt.minute
            df['minute_sin'] = np.sin(2*np.pi*df['minute']/60)
            df['minute_cos'] = np.cos(2*np.pi*df['minute']/60)
            df['second'] = df['timestamp'].dt.second

        # =======================================================
        # NUMERIC CASTING
        # =======================================================
        numeric_cols = [
            'open','high','low','close','volume',
            'quote_asset_volume','number_of_trades',
            'taker_buy_base','taker_buy_quote'
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
        df['rsi_14'] = 100 - (100/(1+rs))

        # =======================================================
        # BOLLINGER BANDS
        # =======================================================
        df['bb_mid'] = df['rolling_mean_20']
        df['bb_std'] = df['rolling_std_20']
        df['bb_upper'] = df['bb_mid'] + 2*df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2*df['bb_std']
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
        # MOMENTUM
        # =======================================================
        for p in [3, 5, 10, 20]:
            df[f'momentum_{p}'] = df['close'] - df['close'].shift(p)

        # =======================================================
        # VWAP
        # =======================================================
        # df["vwap_20"] = (
        #         (df["close"] * df["volume"]).rolling(20).sum()
        #         / df["volume"].rolling(20).sum()
        # )

        # =======================================================
        # PRICE VS MA
        # =======================================================
        for w in [5, 10, 20]:
            df[f'close_vs_ma_{w}'] = df['close'] / df[f'rolling_mean_{w}'] - 1

        # =======================================================
        # 3-CLASS TARGET (target = -1, 0, 1)
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
        df.drop(['timestamp','open_time','close_time'], axis=1, inplace=True)
        df.drop(columns=["future_close", "future_return"], inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # =======================================================
        # SAVE
        # =======================================================
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(DATA_DIR, base + "_feature.csv")
        df.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")


