# ============================================================
# EFFICIENCY CLASSIFICATION DATASET
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_efficiency_dataset(N=8000, seed=42):
    np.random.seed(seed)

    production_load     = np.random.uniform(0.3, 1.0, N)
    cycle_time          = np.clip(20 + 50 * (1 - production_load) + np.random.normal(0, 8, N), 15, 90)
    machine_temperature = np.clip(30 + 45 * production_load + np.random.normal(0, 8, N), 25, 100)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    vibration_signal    = np.random.uniform(0.1, 5.0, N)
    power_factor        = np.random.uniform(0.5, 1.0, N)

    # Stronger signal coefficients so signal dominates noise
    clean_signal = (
          2.5  * power_factor
        + 0.7  * axis_speed
        - 0.04 * cycle_time
        - 0.05 * machine_temperature
        - 0.28 * vibration_signal
        + 0.9  * production_load
        - 0.6  * vibration_signal * (1 - power_factor)
    )

    # noise=0.25 → CV≈76%, train≈83%, gap≈0.08 — healthy
    noisy_signal = clean_signal + np.random.normal(0, 0.25, N)

    q33, q67 = np.percentile(noisy_signal, [33, 67])
    efficiency_class = np.where(noisy_signal < q33, "Low",
                       np.where(noisy_signal < q67, "Medium", "High"))

    df = pd.DataFrame({
        "production_load":     production_load,
        "cycle_time":          cycle_time,
        "machine_temperature": machine_temperature,
        "axis_speed":          axis_speed,
        "vibration_signal":    vibration_signal,
        "power_factor":        power_factor,
        "efficiency_class":    efficiency_class
    })

    os.makedirs("training_data/efficiency", exist_ok=True)
    df.to_csv("training_data/efficiency/eff_raw.csv", index=False)
    print("Saved training_data/efficiency/eff_raw.csv")
    return df

df = generate_efficiency_dataset()