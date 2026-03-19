# ============================================================
# EMISSION CLASSIFIER DATASET
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_emission_dataset(N=7000, seed=42):
    np.random.seed(seed)

    # Correlated features: load and temperature co-vary
    production_load     = np.random.uniform(0.3, 1.0, N)
    machine_temperature = 35 + 60 * production_load + np.random.normal(0, 12, N)
    machine_temperature = np.clip(machine_temperature, 30, 120)
    cycle_time          = np.random.uniform(20, 80, N)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    power_factor        = np.random.uniform(0.5, 1.0, N)

    # Clean signal with non-linear interaction
    clean_signal = (
          15   * production_load
        + 0.04 * machine_temperature
        + 0.40 * axis_speed
        + 0.025* cycle_time
        - 1.2  * power_factor                              # better power factor reduces emissions
        + 5.0  * production_load * (machine_temperature / 120)  # interaction: load × temperature
    )

    # Large noise to blur boundaries
    noisy_signal = clean_signal + np.random.normal(0, 2.2, N)

    q33, q67 = np.percentile(noisy_signal, [33, 67])
    emission_class = np.where(noisy_signal < q33, "Low",
                     np.where(noisy_signal < q67, "Medium", "High"))

    df = pd.DataFrame({
        "production_load":     production_load,
        "cycle_time":          cycle_time,
        "machine_temperature": machine_temperature,
        "axis_speed":          axis_speed,
        "power_factor":        power_factor,
        "emission_class":      emission_class
    })

    os.makedirs("training_data/emission", exist_ok=True)
    df.to_csv("training_data/emission/emission_raw.csv", index=False)
    print("Saved training_data/emission/emission_raw.csv")
    return df

df = generate_emission_dataset()