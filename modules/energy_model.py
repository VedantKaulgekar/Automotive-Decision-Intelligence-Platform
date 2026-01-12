import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/energy_model.pkl"
META_PATH = "models/energy_metadata.pkl"


class EnergyModel:
    def __init__(self):

        # Load trained model
        self.model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

        # Load metadata (feature order + means)
        if os.path.exists(META_PATH):
            meta = joblib.load(META_PATH)
            self.feature_order = meta["feature_order"]
            self.feature_means = meta["feature_means"]
        else:
            self.feature_order = []
            self.feature_means = {}

    # ----------------------------------------------------------
    # Build the exact feature vector required by trained model
    # ----------------------------------------------------------
    def _prepare_features(self, load, cycle, temp, velocity):
        """
        Dashboard user provides ONLY 4 values:
        load, cycle, temp, velocity

        Missing raw features are filled using training MEAN values.
        """

        # Base row with means
        row = {feat: self.feature_means.get(feat, 0) for feat in self.feature_order}

        # Overwrite available ones
        row.update({
            "production_load": load,
            "cycle_time": cycle,
            "machine_temperature": temp,
            "axis_speed": velocity,
        })

        df = pd.DataFrame([row])

        # --- ENGINEERED FEATURES RECREATION ---
        if "load_squared" in self.feature_order:
            df["load_squared"] = df["production_load"] ** 2

        if "temperature_deviation" in self.feature_order:
            df["temperature_deviation"] = (
                df["machine_temperature"] - self.feature_means.get("machine_temperature", 0)
            )

        if "speed_per_load" in self.feature_order:
            df["speed_per_load"] = df["axis_speed"] / (df["production_load"] + 1e-6)

        # Ensure correct column order
        df = df[self.feature_order]

        return df.values.astype(float)

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    def predict(self, load, cycle, temp, velocity):
        if not self.model:
            raise FileNotFoundError("Energy model not trained yet!")

        X = self._prepare_features(load, cycle, temp, velocity)
        pred = float(self.model.predict(X)[0])

        return max(pred, 0.01)
