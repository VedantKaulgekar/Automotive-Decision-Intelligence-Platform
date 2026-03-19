# ============================================================
# EFFICIENCY CLASSIFICATION DATASET
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_efficiency_dataset(N=8000, seed=42):
    np.random.seed(seed)

    # Correlated features
    production_load     = np.random.uniform(0.3, 1.0, N)
    cycle_time          = 20 + 50 * (1 - production_load) + np.random.normal(0, 8, N)  # shorter cycles at higher load
    cycle_time          = np.clip(cycle_time, 15, 90)
    machine_temperature = 30 + 45 * production_load + np.random.normal(0, 8, N)
    machine_temperature = np.clip(machine_temperature, 25, 100)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    vibration_signal    = np.random.uniform(0.1, 5.0, N)
    power_factor        = np.random.uniform(0.5, 1.0, N)

    # Clean signal for boundary placement
    clean_signal = (
          1.8  * power_factor
        + 0.5  * axis_speed
        - 0.03 * cycle_time
        - 0.04 * machine_temperature
        - 0.20 * vibration_signal
        + 0.6  * production_load
        - 0.5  * vibration_signal * (1 - power_factor)     # interaction: vibration hurts more at low pf
    )

    # Apply generous noise BEFORE cutting boundaries — blurs decision boundary
    noisy_signal = clean_signal + np.random.normal(0, 1.2, N)

    # Use quantile-based cuts so classes are balanced despite noise
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