DATA_DIR = "data"

# Each dataset: filename, start date, interval
FILES = {
    "daily": {
        "filename": "btc_daily.csv",
        "start": "2010-01-01",
        "interval": "1d"
    },
    "hourly": {
        "filename": "btc_hourly_5y.csv",
        "start": "5y",   # 5 years ago
        "interval": "1h"
    },
    "minute": {
        "filename": "btc_minute_6m.csv",
        "start": "6m",   # 6 months ago
        "interval": "1m"
    },
}

BASE_URL = "https://api.binance.com/api/v3/klines"