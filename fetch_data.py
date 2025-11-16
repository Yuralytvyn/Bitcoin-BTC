import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict

DATA_DIR = "data"

FILES: Dict[str, str] = {
    "daily": "btc_daily.csv",
    "hourly": "btc_hourly_5y.csv",
    "minute": "btc_minute_6m.csv",
}

BASE_URL = "https://api.binance.com/api/v3/klines"


def ensure_data_dir() -> None:
    """Create the data directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[INFO] Created directory: {DATA_DIR}")


def file_exists(name: str) -> bool:
    """Check if a file exists and is not empty."""
    path = os.path.join(DATA_DIR, FILES[name])
    return os.path.exists(path) and os.path.getsize(path) > 1000


def fetch_data(start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Fetch full OHLCV data for BTC/USDT from Binance."""
    print(f"[FETCH] {interval} data from {start.date()} to {end.date()}")
    all_data = []
    limit = 1000  # Binance API allows 1000 candles per request

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

    # Convert datatypes
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
    path = os.path.join(DATA_DIR, FILES[name])
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(df)} rows)")


def load_csv(name: str) -> pd.DataFrame:
    """Load a CSV file as a pandas DataFrame."""
    path = os.path.join(DATA_DIR, FILES[name])
    return pd.read_csv(path, parse_dates=["timestamp"])


def check_and_update_data(name: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Ensure the dataset exists and is up-to-date."""
    ensure_data_dir()

    intervals: Dict[str, str] = {
        "daily": "1d",
        "hourly": "1h",
        "minute": "1m",
    }

    if not file_exists(name):
        print(f"[INFO] No existing {name} data — fetching from scratch")
        df = fetch_data(start, end, intervals[name])
        save_csv(df, name)
        return df

    df = load_csv(name)
    last_date = df["timestamp"].max().to_pydatetime()

    if last_date < end - timedelta(minutes=5):
        print(f"[UPDATE] Updating {name} from {last_date} to {end}")
        new_df = fetch_data(last_date, end, intervals[name])
        df = pd.concat([df, new_df]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        save_csv(df, name)
    else:
        print(f"[OK] {name} data is up-to-date")

    return df


def prepare_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Create or update all BTC datasets (daily, hourly, minute).
    Returns a dict with all DataFrames.
    """
    ensure_data_dir()
    now = datetime.utcnow()
    datasets = {
        "daily": (datetime(2010, 1, 1), now),
        "hourly": (now - timedelta(days=5 * 365), now),
        "minute": (now - timedelta(days=180), now),
    }

    btc_data = {}
    for name, (start, end) in datasets.items():
        btc_data[name] = check_and_update_data(name, start, end)

    print("\n✅ Усі дані готові:")
    for name, df in btc_data.items():
        print(f"{name}: {len(df)} рядків (від {df['timestamp'].min()} до {df['timestamp'].max()})")

    return btc_data
