# ============================================================
# ENERGY PREDICTION: Regression Dataset
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_energy_dataset(N=10000, seed=42):
    np.random.seed(seed)

    # Correlated features: high load tends to raise temperature
    production_load     = np.random.uniform(0.3, 1.0, N)
    machine_temperature = 30 + 55 * production_load + np.random.normal(0, 10, N)
    machine_temperature = np.clip(machine_temperature, 25, 110)

    cycle_time          = np.random.uniform(20, 80, N)
    axis_speed          = np.random.uniform(0.2, 3.0, N)
    tool_wear           = np.random.beta(2, 3, N)           # realistic skew toward lower wear
    ambient_humidity    = np.random.uniform(20, 90, N)
    vibration_level     = 0.5 + 1.5 * tool_wear + np.random.uniform(0, 2.0, N)  # wear drives vibration
    power_factor        = np.random.uniform(0.5, 1.0, N)

    # Non-linear interaction terms + realistic noise
    signal = (
        10
        + 20  * production_load
        + 3.5 * production_load ** 2                        # quadratic load effect
        + 0.10 * machine_temperature
        + 0.09 * cycle_time
        + 1.4  * axis_speed
        + 2.0  * tool_wear
        + 0.8  * vibration_level
        - 1.5  * power_factor                               # better power factor = less waste
        + 4.0  * production_load * tool_wear                # interaction: worn tool at high load
        + 0.04 * machine_temperature * production_load      # interaction: hot machine at high load
    )

    # Heteroscedastic noise: more variance at higher energy levels (realistic)
    noise_scale = 0.12 * signal
    energy = signal + np.random.normal(0, noise_scale, N)

    df = pd.DataFrame({
        "production_load":     production_load,
        "cycle_time":          cycle_time,
        "machine_temperature": machine_temperature,
        "axis_speed":          axis_speed,
        "tool_wear":           tool_wear,
        "ambient_humidity":    ambient_humidity,
        "vibration_level":     vibration_level,
        "power_factor":        power_factor,
        "energy":              np.clip(energy, 5, None)
    })

    os.makedirs("training_data/energy", exist_ok=True)
    df.to_csv("training_data/energy/energy_raw.csv", index=False)
    print("Saved training_data/energy/energy_raw.csv")
    return df

df = generate_energy_dataset()