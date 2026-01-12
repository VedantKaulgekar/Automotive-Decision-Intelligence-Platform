# ============================================================
# ENERGY PREDICTION: Regression Dataset
# ============================================================
import numpy as np
import pandas as pd
import os

def generate_energy_dataset(N=10000, seed=42):
    np.random.seed(seed)

    production_load     = np.random.uniform(0.3, 1.0, N)
    cycle_time          = np.random.uniform(20, 80, N)
    machine_temperature = np.random.uniform(25, 90, N)
    axis_speed          = np.random.uniform(0.2, 3.0, N)

    tool_wear           = np.random.uniform(0, 1, N)
    ambient_humidity    = np.random.uniform(20, 90, N)
    vibration_level     = np.random.uniform(0.2, 5.0, N)
    power_factor        = np.random.uniform(0.5, 1.0, N)

    # realistic synthetic energy model
    energy = (
        12
        + 24 * production_load
        + 0.08 * machine_temperature
        + 0.12 * cycle_time
        + 1.7 * axis_speed
        + 1.5 * tool_wear
        + np.random.normal(0, 1.4, N)
    )

    df = pd.DataFrame({
        "production_load": production_load,
        "cycle_time": cycle_time,
        "machine_temperature": machine_temperature,
        "axis_speed": axis_speed,
        "tool_wear": tool_wear,
        "ambient_humidity": ambient_humidity,
        "vibration_level": vibration_level,
        "power_factor": power_factor,
        "energy": np.clip(energy, 5, None)
    })

    # Ensure folder exists
    os.makedirs("training_data/energy", exist_ok=True)

    # Save where dashboard expects it
    df.to_csv("training_data/energy/energy_raw.csv", index=False)
    print("Saved training_data/energy/energy_raw.csv")

    return df

df = generate_energy_dataset()
