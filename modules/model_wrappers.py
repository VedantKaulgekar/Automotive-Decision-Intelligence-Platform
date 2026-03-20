"""
model_wrappers.py

Thin wrappers around stored sklearn Pipeline objects.

predict() and predict_batch() now accept **kwargs to override any RAW_DEFAULT
value from the Dashboard's advanced parameter sliders. This means every
feature the model was trained on can be controlled — the 4 primary sliders
handle the main inputs, and the expander sliders override the defaults for
the remaining features.
"""

import numpy as np
import pandas as pd
import streamlit as st
from modules.feature_engineering import add_automotive_features


class BaseWrapper:
    RAW_DEFAULTS = {}
    MODEL_KEY    = ""
    META_KEY     = ""

    def __init__(self):
        self.model         = st.session_state.get(self.MODEL_KEY)
        self.meta          = st.session_state.get(self.META_KEY, {})
        self.feature_order = self.meta.get("feature_order", [])

    def _build_df(self, loads, cycles, temps, speeds, overrides=None):
        """
        Build a feature DataFrame for N samples.

        overrides : dict of column_name → scalar value
                    Any key in overrides replaces the corresponding RAW_DEFAULT
                    for this call. Used by the Dashboard advanced sliders.
        """
        n = len(np.atleast_1d(loads))

        # Start from defaults, apply any overrides on top
        defaults = dict(self.RAW_DEFAULTS)
        if overrides:
            defaults.update(overrides)

        df = pd.DataFrame({
            "production_load":     np.full(n, loads)  if np.isscalar(loads)  else np.asarray(loads,  float),
            "cycle_time":          np.full(n, cycles) if np.isscalar(cycles) else np.asarray(cycles, float),
            "machine_temperature": np.full(n, temps)  if np.isscalar(temps)  else np.asarray(temps,  float),
            "axis_speed":          np.full(n, speeds) if np.isscalar(speeds) else np.asarray(speeds, float),
            **{k: np.full(n, v, dtype=float) for k, v in defaults.items()}
        })

        df = add_automotive_features(df)

        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0

        if self.feature_order:
            return df[self.feature_order]
        return df

    def predict(self, load, cycle, temp, speed, **kwargs):
        """
        Single-sample prediction.
        Any keyword argument matching a RAW_DEFAULT key overrides that default.
        e.g. energy_model.predict(0.8, 40, 60, 1.2, tool_wear=0.8, power_factor=0.85)
        """
        if self.model is None:
            return None
        df = self._build_df(load, cycle, temp, speed, overrides=kwargs or None)
        return self.model.predict(df)[0]

    def predict_batch(self, loads, cycles, temps, speeds, **kwargs):
        """
        Vectorised batch prediction — single Pipeline.predict call for N samples.
        kwargs override RAW_DEFAULTS for the entire batch (scalar values only).
        """
        if self.model is None:
            return None
        df = self._build_df(loads, cycles, temps, speeds, overrides=kwargs or None)
        return self.model.predict(df)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL WRAPPERS
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

    def predict(self, load, cycle, temp, speed, **kwargs):
        pred = super().predict(load, cycle, temp, speed, **kwargs)
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