"""
model_wrappers.py

Thin wrappers around stored sklearn Pipeline objects.
Each wrapper knows which session_state key holds its model and what
default values to use for features not exposed on the Dashboard sliders.

The Pipeline stored in session_state already contains:
  imputer → variance_threshold → robust_scaler → model

So _build_row just needs to produce a raw DataFrame with the right columns
in the right order — the Pipeline handles imputation and scaling internally.
"""

import numpy as np
import pandas as pd
import streamlit as st
from modules.feature_engineering import add_automotive_features


# ─────────────────────────────────────────────────────────────────────────────
# BASE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class BaseWrapper:
    RAW_DEFAULTS = {}
    MODEL_KEY    = ""
    META_KEY     = ""

    def __init__(self):
        self.model        = st.session_state.get(self.MODEL_KEY)
        self.meta         = st.session_state.get(self.META_KEY, {})
        self.feature_order = self.meta.get("feature_order", [])

    def _build_df(self, loads, cycles, temps, speeds):
        """
        Build a DataFrame of N rows with raw sensor values + defaults,
        then apply the same feature engineering used during training.
        The Pipeline (stored inside self.model) handles imputation and scaling.
        """
        n = len(np.atleast_1d(loads))
        df = pd.DataFrame({
            "production_load":     np.full(n, loads)  if np.isscalar(loads)  else np.asarray(loads,  float),
            "cycle_time":          np.full(n, cycles) if np.isscalar(cycles) else np.asarray(cycles, float),
            "machine_temperature": np.full(n, temps)  if np.isscalar(temps)  else np.asarray(temps,  float),
            "axis_speed":          np.full(n, speeds) if np.isscalar(speeds) else np.asarray(speeds, float),
            **{k: np.full(n, v, dtype=float) for k, v in self.RAW_DEFAULTS.items()}
        })

        # Apply identical feature engineering as during training
        df = add_automotive_features(df)

        # Ensure all columns the model was trained on are present
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0

        # Return only the columns in the order the model expects
        if self.feature_order:
            return df[self.feature_order]
        return df

    def predict(self, load, cycle, temp, speed):
        """Single-sample prediction."""
        if self.model is None:
            return None
        df = self._build_df(load, cycle, temp, speed)
        return self.model.predict(df)[0]

    def predict_batch(self, loads, cycles, temps, speeds):
        """
        Vectorised batch prediction — builds one DataFrame for N samples
        and calls Pipeline.predict once.
        Orders of magnitude faster than calling predict() in a Python loop.

        Parameters: 1-D array-like of length N each.
        Returns: np.ndarray of shape (N,)
        """
        if self.model is None:
            return None
        df = self._build_df(loads, cycles, temps, speeds)
        return self.model.predict(df)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL MODEL WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────
class EnergyModel(BaseWrapper):
    RAW_DEFAULTS = {
        "tool_wear":        0.5,
        "ambient_humidity": 50.0,
        "vibration_level":  1.0,
        "power_factor":     0.9,
    }
    MODEL_KEY = "energy_model"
    META_KEY  = "energy_metadata"

    def predict(self, load, cycle, temp, speed):
        pred = super().predict(load, cycle, temp, speed)
        if pred is None:
            return None
        return max(float(pred), 0.01)


class EfficiencyModel(BaseWrapper):
    RAW_DEFAULTS = {
        "vibration_signal": 1.0,
        "power_factor":     0.9,
    }
    MODEL_KEY = "efficiency_model"
    META_KEY  = "efficiency_metadata"


class EmissionModel(BaseWrapper):
    RAW_DEFAULTS = {
        "power_factor": 0.9,
    }
    MODEL_KEY = "emission_model"
    META_KEY  = "emission_metadata"


class MaintenanceModel(BaseWrapper):
    RAW_DEFAULTS = {
        "vibration_level": 1.0,
        "tool_wear":       0.5,
        "oil_quality":     0.7,
        "pressure":        100.0,
    }
    MODEL_KEY = "maintenance_model"
    META_KEY  = "maintenance_metadata"