# feature_engineering.py
import pandas as pd
import numpy as np

def add_automotive_features(df):
    df = df.copy()

    # Safe feature engineering for regression & classifiers
    if "production_load" in df.columns:
        df["load_squared"] = df["production_load"] ** 2

    if "machine_temperature" in df.columns:
        df["temperature_deviation"] = (
            df["machine_temperature"] - df["machine_temperature"].mean()
        )

    if "axis_speed" in df.columns and "production_load" in df.columns:
        df["speed_per_load"] = df["axis_speed"] / (df["production_load"] + 1e-6)

    return df
