# ============================================================
# EFFICIENCY CLASSIFICATION DATASET
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_efficiency_dataset(N=8000, seed=42):
    np.random.seed(seed)

    production_load     = np.random.uniform(0.3, 1.0, N)
    cycle_time          = np.random.uniform(15, 90, N)
    machine_temperature = np.random.uniform(25, 100, N)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    vibration_signal    = np.random.uniform(0.1, 5.0, N)
    power_factor        = np.random.uniform(0.5, 1.0, N)

    eff_score = (
        + 1.5 * power_factor
        + 0.4 * axis_speed
        - 0.04 * cycle_time
        - 0.05 * machine_temperature
        - 0.25 * vibration_signal
        + np.random.normal(0, 0.6, N)
    )

    labels = pd.cut(eff_score, bins=3, labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "production_load": production_load,
        "cycle_time": cycle_time,
        "machine_temperature": machine_temperature,
        "axis_speed": axis_speed,
        "vibration_signal": vibration_signal,
        "power_factor": power_factor,
        "efficiency_class": labels
    })

    os.makedirs("training_data/efficiency", exist_ok=True)
    df.to_csv("training_data/efficiency/eff_raw.csv", index=False)
    print("Saved training_data/efficiency/eff_raw.csv")

    return df

df = generate_efficiency_dataset()
