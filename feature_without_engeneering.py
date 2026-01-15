
import pandas as pd
import numpy as np

def compute_target(df: pd.DataFrame, threshold=0.002) -> pd.Series:
    """Повертає Series з класами 0/1/2 для переданого DataFrame."""
    ret = df["close"].pct_change().shift(-1)
    return pd.cut(
        ret,
        bins=[-999, -threshold, threshold, 999],
        labels=[0, 1, 2]
    )

def generate_target(file_path: str, threshold=0.002) -> str:
    """Читає CSV, додає target, зберігає *_feature.csv і повертає шлях."""
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["target"] = compute_target(df, threshold)

    df_final = df.dropna().reset_index(drop=True)
    output_path = file_path.replace(".csv", "_feature.csv")
    df_final.to_csv(output_path, index=False)

    print(f"\n🎉 CLEAN DATASET READY → {df_final.shape}")
    print(f"📁 Saved to {output_path}\n")
    return output_path