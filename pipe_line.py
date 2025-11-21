from datetime import datetime, timedelta
from fetch_data import fetch_data, save_csv
from feature_engineering import *
from split_train_test import *
from Model import *
from init import *
# ---- Ensure data directory exists ----
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print("Created directory:", DATA_DIR)

# ---- Check each CSV and create if missing ----
for name, info in FILES.items():
    csv_path = os.path.join(DATA_DIR, info["filename"])

    if not os.path.isfile(csv_path):
        print(f"{info['filename']} does not exist — creating...")

        # Determine start date
        if info["start"] == "5y":
            start = datetime.utcnow() - timedelta(days=5*365)
        elif info["start"] == "6m":
            start = datetime.utcnow() - timedelta(days=180)
        else:
            start = datetime.strptime(info["start"], "%Y-%m-%d")

        end = datetime.utcnow()
        interval = info["interval"]

        df = fetch_data(start, end, interval)
        save_csv(df, name)

    else:
        print(f"{info['filename']} already exists — skipping.")

if not os.path.exists("btc_hourly_5y_feature.csv"):
    daily_hourly_minute()
if not (os.path.exists("btc_hourly_5y_feature_test.csv") and os.path.exists("btc_hourly_5y_feature_train")):
    train_test_split()
start_model()
