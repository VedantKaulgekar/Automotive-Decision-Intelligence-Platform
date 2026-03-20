"""
feature_engineering.py — Automotive Decision Intelligence Platform

Derives physically meaningful features from raw sensor readings.
All new columns are grounded in manufacturing domain knowledge.

Feature list
------------
Existing (retained):
  load_squared          — non-linear quadratic load effect on energy
  temperature_deviation — centred temperature (relative to dataset mean)
  speed_per_load        — axis speed normalised by production load

New (added):
  thermal_load          — production_load × machine_temperature
                          Primary combined driver of energy and emissions.
                          High load at high temperature is far worse than either alone.

  mechanical_stress     — vibration_level × axis_speed
                          Vibration at high speed accelerates component wear.
                          Key predictor for maintenance risk.

  wear_index            — tool_wear × vibration_level
                          Combined degradation signal. Both individually indicate
                          wear; together they indicate imminent failure.

  effective_power       — production_load × axis_speed × power_factor
                          Approximates actual mechanical power output.
                          Accounts for electrical efficiency (power factor).

All features are computed with safe guards (column existence checks, epsilon
for division) so the function works across all 4 dataset schemas without
needing different versions per model.
"""

import pandas as pd
import numpy as np


def add_automotive_features(df):
    """
    Add engineered features to a DataFrame. Safe to call on any of the
    4 model datasets — columns that don't exist are simply skipped.

    Parameters
    ----------
    df : pd.DataFrame  — after imputation, before scaling

    Returns
    -------
    pd.DataFrame with additional derived columns appended
    """
    df = df.copy()

    # ── Existing features (unchanged) ────────────────────────────────────────

    if "production_load" in df.columns:
        df["load_squared"] = df["production_load"] ** 2

    if "machine_temperature" in df.columns:
        df["temperature_deviation"] = (
            df["machine_temperature"] - df["machine_temperature"].mean()
        )

    if "axis_speed" in df.columns and "production_load" in df.columns:
        df["speed_per_load"] = df["axis_speed"] / (df["production_load"] + 1e-6)

    # ── New interaction features ──────────────────────────────────────────────

    # Thermal load: combined heat generation under load
    # High load at high temperature is the dominant driver of both energy
    # consumption and emission class — this captures the synergistic effect.
    if "production_load" in df.columns and "machine_temperature" in df.columns:
        df["thermal_load"] = df["production_load"] * df["machine_temperature"]

    # Mechanical stress: vibration amplified by speed
    # Vibration at high speed causes disproportionately more wear than at
    # low speed. Used primarily by the Maintenance classifier.
    if "vibration_level" in df.columns and "axis_speed" in df.columns:
        df["mechanical_stress"] = df["vibration_level"] * df["axis_speed"]

    # Also covers the efficiency dataset which uses vibration_signal
    if "vibration_signal" in df.columns and "axis_speed" in df.columns:
        df["mechanical_stress"] = df["vibration_signal"] * df["axis_speed"]

    # Wear index: combined degradation indicator
    # Both tool wear and vibration individually indicate mechanical degradation.
    # Their product creates a high-value signal specifically for imminent
    # maintenance need — low on both is fine, high on either is concerning,
    # high on both is critical.
    if "tool_wear" in df.columns and "vibration_level" in df.columns:
        df["wear_index"] = df["tool_wear"] * df["vibration_level"]

    # Effective power: load × speed × power factor
    # Approximates actual mechanical power output accounting for electrical
    # efficiency. A machine running at high load with poor power factor
    # wastes more energy than the load alone suggests.
    if all(c in df.columns for c in
           ["production_load", "axis_speed", "power_factor"]):
        df["effective_power"] = (
            df["production_load"] * df["axis_speed"] * df["power_factor"]
        )

    return df