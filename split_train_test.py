import pandas as pd
import os

from init import *

def train_test_split():
    # Вкажи тут назву файлу
    FILENAME = "btc_hourly_5y_feature.csv"

    path = os.path.join(DATA_DIR, FILENAME)
    df = pd.read_csv(path)

    TEST_SIZE = 0.2   # 20% у тест

    # таймсіріс спліт
    n = len(df)
    split_idx = int(n * (1 - TEST_SIZE))

    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    # нові назви
    base = FILENAME.replace(".csv", "")
    train_path = os.path.join(DATA_DIR, base + "_train.csv")
    test_path  = os.path.join(DATA_DIR, base + "_test.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")