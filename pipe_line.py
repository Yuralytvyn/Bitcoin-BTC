from datetime import datetime, timezone, timedelta
import os

from fetch_data import fetch_data, save_csv
from feature_without_engeneering import generate_target
from feature_engineering import daily_hourly_minute
from split_train_test import train_test_split
from Model import start_model
from init import DATA_DIR, FILES


def main():

    # ---- Ensure data directory exists ----
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Created directory:", DATA_DIR)

    # ---- Check each CSV and create if missing ----
    # As for now only 1 file
    for name, info in FILES.items():
        csv_path = os.path.join(DATA_DIR, info["filename"])

        if not os.path.isfile(csv_path):
            print(f"{info['filename']} does not exist — creating...")

            if info["start"] == "5y":
                start = datetime.now(timezone.utc) - timedelta(days=5 * 365)
            end = datetime.now(timezone.utc)
            interval = info["interval"]

            df = fetch_data(start, end, interval)
            save_csv(df, name)
        else:
            print(f"{info['filename']} already exists — skipping.")

    # FEATURE ENGINEERING
    feature_file = os.path.join(DATA_DIR, "btc_hourly_5y_feature.csv")
    if not os.path.exists(feature_file):
        hourly_file = os.path.join(DATA_DIR, "btc_hourly_5y.csv")
        daily_hourly_minute([hourly_file])

    # TRAIN/TEST SPLIT
    train_path = os.path.join(DATA_DIR, "btc_hourly_5y_feature_train.csv")
    test_path  = os.path.join(DATA_DIR, "btc_hourly_5y_feature_test.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        train_test_split()

    # MODEL TRAINING
    start_model()


if __name__ == "__main__":
    main()
