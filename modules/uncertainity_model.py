import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR UNCERTAINTY SPECIFICATIONS
# Realistic measurement uncertainty for each input sensor:
#
#   production_load    : ±3%  of reading  — load cell / current transducer
#   cycle_time         : ±0.5 s absolute  — PLC timer resolution
#   machine_temperature: ±2°C absolute    — thermocouple (Type K) accuracy
#   axis_speed         : ±2%  of reading  — rotary encoder uncertainty
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_UNCERTAINTY = {
    "production_load":      {"type": "relative", "std_frac": 0.03},
    "cycle_time":           {"type": "absolute", "std_abs":  0.5 },
    "machine_temperature":  {"type": "absolute", "std_abs":  2.0 },
    "axis_speed":           {"type": "relative", "std_frac": 0.02},
}

INPUT_BOUNDS = {
    "production_load":      (0.3,  1.0),
    "cycle_time":           (10.0, 90.0),
    "machine_temperature":  (20.0, 120.0),
    "axis_speed":           (0.2,  3.0),
}


def monte_carlo_energy(model, load, cycle, temp, speed, samples=500, seed=None):
    """
    Proper Monte Carlo simulation for energy prediction uncertainty.

    Samples each input from its sensor uncertainty distribution, then runs
    the model ONCE via predict_batch (vectorised) instead of N times in a loop.
    This is orders of magnitude faster than the naive loop approach.

    Returns
    -------
    results      : np.ndarray (samples,) — distribution of energy predictions (kWh)
    sampled_inputs : dict — the sampled input arrays for sensitivity diagnostics
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: Sample all inputs at once (fully vectorised) ──────────────────
    def sample(measured, key):
        spec = SENSOR_UNCERTAINTY[key]
        std  = spec["std_frac"] * abs(measured) if spec["type"] == "relative" else spec["std_abs"]
        arr  = rng.normal(loc=measured, scale=std, size=samples)
        lo, hi = INPUT_BOUNDS[key]
        return np.clip(arr, lo, hi)

    loads  = sample(load,  "production_load")
    cycles = sample(cycle, "cycle_time")
    temps  = sample(temp,  "machine_temperature")
    speeds = sample(speed, "axis_speed")

    # ── Step 2: Single vectorised model call (not a Python loop) ─────────────
    raw = model.predict_batch(loads, cycles, temps, speeds)

    if raw is None:
        # Model not trained yet — return zeros
        results = np.full(samples, 0.01)
    else:
        results = np.maximum(raw.astype(float), 0.01)

    sampled_inputs = {
        "production_load":     loads,
        "cycle_time":          cycles,
        "machine_temperature": temps,
        "axis_speed":          speeds,
    }

    return results, sampled_inputs