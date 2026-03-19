# ============================================================
# PREDICTIVE MAINTENANCE DATASET
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_maintenance_dataset(N=9000, seed=42):
    np.random.seed(seed)

    tool_wear           = np.random.beta(2, 4, N)
    vibration_level     = np.clip(0.5 + 4.0 * tool_wear + np.random.uniform(0, 1.5, N), 0.1, 6.0)
    production_load     = np.random.uniform(0.3, 1.0, N)
    machine_temperature = np.clip(35 + 50 * production_load + 10 * tool_wear + np.random.normal(0, 10, N), 30, 130)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    oil_quality         = np.clip(1.0 - 0.6 * tool_wear + np.random.normal(0, 0.15, N), 0.1, 1.0)
    pressure            = np.random.uniform(60, 150, N)

    clean_risk = (
          0.70 * vibration_level
        + 0.04 * machine_temperature
        + 1.20 * tool_wear
        - 1.00 * oil_quality
        + 0.015* pressure
        + 0.30 * production_load
        + 1.50 * tool_wear * vibration_level
        - 0.80 * oil_quality * (1 - tool_wear)
    )

    # noise=0.20 → CV≈90%, train≈94%, gap≈0.03 — excellent
    noisy_risk = clean_risk + np.random.normal(0, 0.20, N)

    q33, q67 = np.percentile(noisy_risk, [33, 67])
    maintenance_class = np.where(noisy_risk < q33, "Low",
                        np.where(noisy_risk < q67, "Medium", "High"))

    df = pd.DataFrame({
        "production_load":     production_load,
        "machine_temperature": machine_temperature,
        "axis_speed":          axis_speed,
        "vibration_level":     vibration_level,
        "tool_wear":           tool_wear,
        "oil_quality":         oil_quality,
        "pressure":            pressure,
        "maintenance_class":   maintenance_class
    })

    os.makedirs("training_data/maintenance", exist_ok=True)
    df.to_csv("training_data/maintenance/maintenance_raw.csv", index=False)
    print("Saved training_data/maintenance/maintenance_raw.csv")
    return df

df = generate_maintenance_dataset()