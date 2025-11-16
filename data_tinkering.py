import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Tuple

DATA_DIR = "data"

FILES: Dict[str, str] = {
    "daily": "btc_daily.csv",
    "hourly": "btc_hourly_5y.csv",
    "minute": "btc_minute_6m.csv",
}


def ensure_data_dir() -> None:
    """Create data directory if not exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[INFO] Created folder {DATA_DIR}")


def file_exists(name: str) -> bool:
    """Check if the CSV file exists and is not empty."""
    path = os.path.join(DATA_DIR, FILES[name])
    return os.path.exists(path) and os.path.getsize(path) > 1000


def fetch_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch BTC/USD data from CoinGecko API within a given time range."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": int(start.timestamp()),
        "to": int(end.timestamp()),
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    market_caps = data.get("market_caps", [])

    df = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcfromtimestamp(p[0] / 1000),
                "price": p[1],
                "volume": v[1],
                "market_cap": m[1],
            }
            for p, v, m in zip(prices, volumes, market_caps)
        ]
    )
    return enrich_features(df)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features such as lags, percentage change, rolling stats."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Price and volume changes
    df["price_change"] = df["price"].diff()
    df["price_pct_change"] = df["price"].pct_change() * 100
    df["volume_change"] = df["volume"].diff()

    # Rolling statistics
    df["rolling_mean"] = df["price"].rolling(window=5, min_periods=1).mean()
    df["rolling_std"] = df["price"].rolling(window=5, min_periods=1).std()

    # Lags (previous prices)
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    # Volatility estimation
    df["volatility"] = df["price_pct_change"].rolling(window=10, min_periods=1).std()

    # Drop NaNs from the beginning if needed
    df = df.dropna().reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame as CSV in data/ directory."""
    path = os.path.join(DATA_DIR, FILES[name])
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(df)} rows)")


def load_csv(name: str) -> pd.DataFrame:
    """Load CSV from data directory."""
    path = os.path.join(DATA_DIR, FILES[name])
    return pd.read_csv(path, parse_dates=["timestamp"])


def check_and_update_data(name: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Check if data exists; if not, fetch it.
    If exists, update with the latest records.
    """
    ensure_data_dir()

    if not file_exists(name):
        print(f"[INFO] No {name} data found — fetching from scratch.")
        df = fetch_data(start, end)
        save_csv(df, name)
        return df

    df = load_csv(name)
    last_date = df["timestamp"].max().to_pydatetime()

    if last_date < end - timedelta(hours=1):
        print(f"[UPDATE] Updating {name} from {last_date} to {end}")
        new_df = fetch_data(last_date, end)
        df = pd.concat([df, new_df]).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        save_csv(df, name)
    else:
        print(f"[OK] {name} data is up to date")

    return df
