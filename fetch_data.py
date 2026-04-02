import os
import pandas as pd
import requests
from datetime import datetime
from init import *

# =========================
# EXTRA API
# =========================

def get_usd_rate():
    # якщо хочеш USD курс окремо (наприклад до гривні)
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        return r.json()["rates"]["UAH"]
    except:
        return None


def get_top_btc_addresses():
    # топ адреси BTC (не люди!)
    try:
        r = requests.get("https://blockchain.info/balance?active=1AJbsFZ64EpEfS5UAjAfcUG8pH8Jn3rn1F")
        return r.json()
    except:
        return None


def get_oil_price():
    try:
        r = requests.get("https://api.oilpriceapi.com/v1/prices/latest")
        return r.json()
    except:
        return None


# =========================
# MAIN FETCH
# =========================

def fetch_data(start: datetime, end: datetime, interval: str) -> pd.DataFrame:
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

    # =========================
    # TYPES
    # =========================
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    # =========================
    # 🔥 ТЕ, ЩО ТИ ХОТІВ
    # =========================

    # 1. куплено / продано
    df["buy_volume"] = df["taker_buy_base"]
    df["sell_volume"] = df["volume"] - df["taker_buy_base"]

    # 2. USD курс (UAH для прикладу)
    usd_rate = get_usd_rate()
    df["usd_to_uah"] = usd_rate

    # 3. ціна нафти (одна на весь датасет)
    oil_data = get_oil_price()
    oil_price = None

    if oil_data and "data" in oil_data:
        oil_price = oil_data["data"]["price"]

    df["oil_price"] = oil_price

    # 4. топ BTC адреси (заглушка, бо API нормального нема)
    top_addresses = get_top_btc_addresses()

    # ти не запхаєш це нормально в df → це інший тип даних
    print("[INFO] Top BTC addresses sample:", top_addresses)

    # =========================

    df = df[
        [
            "timestamp", "open", "high", "low", "close",
            "volume", "buy_volume", "sell_volume",
            "quote_asset_volume", "number_of_trades",
            "usd_to_uah", "oil_price"
        ]
    ]

    return df


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(DATA_DIR, FILES[name]["filename"])
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(df)} rows)")