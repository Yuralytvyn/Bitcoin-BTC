import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from init import *

def fetch_data(start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Fetch full OHLCV data for BTC/USDT from Binance."""
    print(f"[FETCH] {interval} data from {start.date()} to {end.date()}")
    all_data = []
    limit = 1000

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while True:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": start_ms,
            "limit": limit,
        }
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        all_data.extend(data)
        last_close_time = data[-1][6]
        if last_close_time >= end_ms:
            break

        start_ms = last_close_time + 1

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
    )

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    df = df[
        [
            "timestamp", "open", "high", "low", "close",
            "volume", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "open_time", "close_time"
        ]
    ]

    return df


def save_csv(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame to a CSV file inside the data directory."""
    path = os.path.join(DATA_DIR, FILES[name]["filename"])
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(df)} rows)")
