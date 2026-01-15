DATA_DIR = "data"

# Each dataset: filename, start date, interval
FILES = {

    "hourly": {
        "filename": "btc_hourly_5y.csv",
        "start": "5y",   # 5 years ago
        "interval": "1h"
    },

}

BASE_URL = "https://api.binance.com/api/v3/klines"

VALIDATION = True
VALIDATION_SIZE = 0.2