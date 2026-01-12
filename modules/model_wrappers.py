import os
import joblib
import numpy as np
import pandas as pd


# -----------------------------
#  SHARED FEATURE ENGINEERING
# -----------------------------
def apply_engineering(df: pd.DataFrame):
    """Applies SAME engineered features used during training."""
    df["load_squared"] = df["production_load"] ** 2
    df["temperature_deviation"] = df["machine_temperature"] - 60
    df["speed_per_load"] = df["axis_speed"] / (df["production_load"] + 1e-6)
    return df


def load_metadata(path):
    """Loads metadata saved by Model_Training_Studio for feature order."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ====================================================
#                 ENERGY MODEL WRAPPER
# ====================================================
class EnergyModel:
    RAW_DEFAULTS = {
        "tool_wear": 0.5,
        "ambient_humidity": 50,
        "vibration_level": 1.0,
        "power_factor": 0.9
    }

    MODEL_PATH = "models/energy_model.pkl"
    META_PATH = "models/energy_metadata.pkl"

    def __init__(self):
        self.model = joblib.load(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None
        self.meta = load_metadata(self.META_PATH) or {}

        # Training feature order
        self.feature_order = self.meta.get("feature_order", [])

    def _build_features(self, load, cycle, temp, speed):

        df = pd.DataFrame([{
            "production_load": load,
            "cycle_time": cycle,
            "machine_temperature": temp,
            "axis_speed": speed,
            **self.RAW_DEFAULTS
        }])

        df = apply_engineering(df)

        # Guarantee all required columns exist
        for col in self.feature_order:
            if col not in df:
                df[col] = 0.0

        return df[self.feature_order].values.astype(float)

    def predict(self, load, cycle, temp, speed):
        if not self.model:
            return None
        X = self._build_features(load, cycle, temp, speed)
        pred = float(self.model.predict(X)[0])
        return max(pred, 0.01)


# ====================================================
#             EFFICIENCY MODEL WRAPPER
# ====================================================
class EfficiencyModel:
    RAW_DEFAULTS = {
        "vibration_signal": 1.0,
        "power_factor": 0.9,
    }

    MODEL_PATH = "models/efficiency_classifier.pkl"
    META_PATH = "models/efficiency_metadata.pkl"

    def __init__(self):
        self.model = joblib.load(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None
        self.meta = load_metadata(self.META_PATH) or {}
        self.feature_order = self.meta.get("feature_order", [])

    def _build_features(self, load, cycle, temp, speed):

        df = pd.DataFrame([{
            "production_load": load,
            "cycle_time": cycle,
            "machine_temperature": temp,
            "axis_speed": speed,
            **self.RAW_DEFAULTS
        }])

        df = apply_engineering(df)

        # add missing cols
        for col in self.feature_order:
            if col not in df:
                df[col] = 0.0

        return df[self.feature_order].values.astype(float)

    def predict(self, load, cycle, temp, speed):
        if not self.model:
            return None
        X = self._build_features(load, cycle, temp, speed)
        return self.model.predict(X)[0]


# ====================================================
#             EMISSION MODEL WRAPPER
# ====================================================
class EmissionModel:
    RAW_DEFAULTS = {
        "power_factor": 0.9
    }

    MODEL_PATH = "models/emission_classifier.pkl"
    META_PATH = "models/emission_metadata.pkl"

    def __init__(self):
        self.model = joblib.load(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None
        self.meta = load_metadata(self.META_PATH) or {}
        self.feature_order = self.meta.get("feature_order", [])

    def _build_features(self, load, cycle, temp, speed):

        df = pd.DataFrame([{
            "production_load": load,
            "cycle_time": cycle,
            "machine_temperature": temp,
            "axis_speed": speed,
            **self.RAW_DEFAULTS
        }])

        df = apply_engineering(df)

        for col in self.feature_order:
            if col not in df:
                df[col] = 0.0

        return df[self.feature_order].values.astype(float)

    def predict(self, load, cycle, temp, speed):
        if not self.model:
            return None
        X = self._build_features(load, cycle, temp, speed)
        return self.model.predict(X)[0]


# ====================================================
#         MAINTENANCE DL MODEL WRAPPER
# ====================================================
class MaintenanceModel:
    RAW_DEFAULTS = {
        "vibration_level": 1.0,
        "tool_wear": 0.5,
        "oil_quality": 0.7,
        "pressure": 100,
    }

    MODEL_PATH = "models/maintenance_dl.pkl"
    META_PATH = "models/maintenance_metadata.pkl"

    def __init__(self):
        self.model = joblib.load(self.MODEL_PATH) if os.path.exists(self.MODEL_PATH) else None
        self.meta = load_metadata(self.META_PATH) or {}
        self.feature_order = self.meta.get("feature_order", [])

    def _build_features(self, load, cycle, temp, speed):

        df = pd.DataFrame([{
            "production_load": load,
            "cycle_time": cycle,
            "machine_temperature": temp,
            "axis_speed": speed,
            **self.RAW_DEFAULTS
        }])

        df = apply_engineering(df)

        for col in self.feature_order:
            if col not in df:
                df[col] = 0.0

        return df[self.feature_order].values.astype(float)

    def predict(self, load, cycle, temp, speed):
        if not self.model:
            return None
        X = self._build_features(load, cycle, temp, speed)
        return self.model.predict(X)[0]
