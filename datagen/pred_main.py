# ============================================================
# PREDICTIVE MAINTENANCE DATASET
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_maintenance_dataset(N=9000, seed=42):
    np.random.seed(seed)

    production_load     = np.random.uniform(0.3, 1.0, N)
    vibration_level     = np.random.uniform(0.1, 6.0, N)
    machine_temperature = np.random.uniform(30, 130, N)
    tool_wear           = np.random.uniform(0, 1, N)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    oil_quality         = np.random.uniform(0.4, 1.0, N)
    pressure            = np.random.uniform(60, 150, N)

    risk_score = (
        0.65 * vibration_level +
        0.05 * machine_temperature +
        1.1  * tool_wear -
        0.9  * oil_quality +
        0.02 * pressure +
        0.35 * production_load +
        np.random.normal(0, 0.4, N)
    )

    labels = pd.cut(risk_score, 3, labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "production_load": production_load,
        "machine_temperature": machine_temperature,
        "axis_speed": axis_speed,
        "vibration_level": vibration_level,
        "tool_wear": tool_wear,
        "oil_quality": oil_quality,
        "pressure": pressure,
        "maintenance_class": labels
    })

    os.makedirs("training_data/maintenance", exist_ok=True)
    df.to_csv("training_data/maintenance/maintenance_raw.csv", index=False)
    print("Saved training_data/maintenance/maintenance_raw.csv")

    return df


df = generate_maintenance_dataset()
