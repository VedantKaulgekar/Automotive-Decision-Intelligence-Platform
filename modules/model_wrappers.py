import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
#  SHARED FEATURE ENGINEERING
# -----------------------------
def apply_engineering(df: pd.DataFrame):
    df["load_squared"] = df["production_load"] ** 2
    df["temperature_deviation"] = df["machine_temperature"] - 60
    df["speed_per_load"] = df["axis_speed"] / (df["production_load"] + 1e-6)
    return df


# ====================================================
#        GENERIC MODEL WRAPPER BASE CLASS
# ====================================================
class BaseWrapper:
    RAW_DEFAULTS = {}
    MODEL_KEY = ""
    META_KEY = ""

    def __init__(self):
        # Load from session_state, NOT DISK
        self.model = st.session_state.get(self.MODEL_KEY)
        self.meta = st.session_state.get(self.META_KEY, {})
        self.feature_order = self.meta.get("feature_order", [])

    def _build_row(self, load, cycle, temp, speed):
        df = pd.DataFrame([{
            "production_load": load,
            "cycle_time": cycle,
            "machine_temperature": temp,
            "axis_speed": speed,
            **self.RAW_DEFAULTS
        }])

        df = apply_engineering(df)

        # Ensure all required columns exist
        for col in self.feature_order:
            if col not in df:
                df[col] = 0.0

        return df[self.feature_order].values.astype(float)

    def predict(self, load, cycle, temp, speed):
        if self.model is None:
            return None
        X = self._build_row(load, cycle, temp, speed)
        return self.model.predict(X)[0]


# ====================================================
#               INDIVIDUAL MODEL WRAPPERS
# ====================================================

class EnergyModel(BaseWrapper):
    RAW_DEFAULTS = {
        "tool_wear": 0.5,
        "ambient_humidity": 50,
        "vibration_level": 1.0,
        "power_factor": 0.9
    }
    MODEL_KEY = "energy_model"
    META_KEY = "energy_metadata"

    def predict(self, load, cycle, temp, speed):
        pred = super().predict(load, cycle, temp, speed)
        if pred is None:
            return None
        return max(float(pred), 0.01)


class EfficiencyModel(BaseWrapper):
    RAW_DEFAULTS = {
        "vibration_signal": 1.0,
        "power_factor": 0.9
    }
    MODEL_KEY = "efficiency_model"
    META_KEY = "efficiency_metadata"


class EmissionModel(BaseWrapper):
    RAW_DEFAULTS = {
        "power_factor": 0.9
    }
    MODEL_KEY = "emission_model"
    META_KEY = "emission_metadata"


class MaintenanceModel(BaseWrapper):
    RAW_DEFAULTS = {
        "vibration_level": 1.0,
        "tool_wear": 0.5,
        "oil_quality": 0.7,
        "pressure": 100
    }
    MODEL_KEY = "maintenance_model"
    META_KEY = "maintenance_metadata"
