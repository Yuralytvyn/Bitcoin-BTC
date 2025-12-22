from datetime import datetime, timedelta
import os

from fetch_data import fetch_data, save_csv
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

    # FEATURE ENGINEERING
    feature_inputs = []

    for name, info in FILES.items():
        raw_file = os.path.join(DATA_DIR, info["filename"])
        feature_file = raw_file.replace(".csv", "_feature.csv")

        if not os.path.exists(feature_file):
            feature_inputs.append(raw_file)

    if feature_inputs:
        daily_hourly_minute(feature_inputs)

    # TRAIN/TEST SPLIT
    need_split = False

    for name, info in FILES.items():
        feature_file = os.path.join(DATA_DIR, info["filename"].replace(".csv", "_feature.csv"))
        train_file = feature_file.replace(".csv", "_train.csv")
        test_file = feature_file.replace(".csv", "_test.csv")

        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            need_split = True

    if need_split:
        train_test_split()

    # MODEL TRAINING
    start_model()


if __name__ == "__main__":
    main()
