import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR UNCERTAINTY SPECIFICATIONS
# These represent realistic measurement uncertainty for each input sensor,
# grounded in typical industrial sensor accuracy specifications:
#
#   production_load    : ±3%  of reading  — load cell / current transducer
#   cycle_time         : ±0.5 s absolute  — PLC timer resolution
#   machine_temperature: ±2°C absolute    — thermocouple (Type K) accuracy
#   axis_speed         : ±2%  of reading  — rotary encoder uncertainty
#
# Each input is modelled as normally distributed around the measured value
# with the above standard deviations. This reflects the fact that any
# sensor reading is an estimate, not the true physical value.
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_UNCERTAINTY = {
    "production_load":      {"type": "relative", "std_frac": 0.03},   # ±3% of reading
    "cycle_time":           {"type": "absolute", "std_abs":  0.5 },   # ±0.5 s
    "machine_temperature":  {"type": "absolute", "std_abs":  2.0 },   # ±2°C
    "axis_speed":           {"type": "relative", "std_frac": 0.02},   # ±2% of reading
}

# Physical bounds — sampled inputs are clipped to these ranges
INPUT_BOUNDS = {
    "production_load":      (0.3,  1.0),
    "cycle_time":           (10.0, 90.0),
    "machine_temperature":  (20.0, 120.0),
    "axis_speed":           (0.2,  3.0),
}


def monte_carlo_energy(model, load, cycle, temp, speed, samples=500, seed=None):
    """
    Proper Monte Carlo simulation for energy prediction uncertainty.

    Instead of adding noise to the output, this function:
      1. Samples each input from its sensor uncertainty distribution
      2. Runs the trained energy model N times with those sampled inputs
      3. Returns the distribution of predicted energy outcomes

    This correctly propagates input measurement uncertainty through
    the model to produce a meaningful output uncertainty distribution.

    Parameters
    ----------
    model   : trained EnergyModel wrapper with a .predict(load, cycle, temp, speed) method
    load    : float — measured production load (fraction 0.3–1.0)
    cycle   : float — measured cycle time (seconds)
    temp    : float — measured machine temperature (°C)
    speed   : float — measured axis speed (m/s)
    samples : int   — number of Monte Carlo iterations (default 500)
    seed    : int or None — random seed for reproducibility

    Returns
    -------
    results : np.ndarray of shape (samples,) — distribution of energy predictions (kWh)
    inputs  : dict — the sampled input arrays for each variable (for diagnostics)
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: Sample uncertain inputs ──────────────────────────────────────
    # Each sensor reading is the true value + measurement error
    # We model each input as N(measured_value, sensor_std)

    def sample_input(measured, key):
        spec = SENSOR_UNCERTAINTY[key]
        if spec["type"] == "relative":
            std = spec["std_frac"] * abs(measured)
        else:
            std = spec["std_abs"]
        samples_arr = rng.normal(loc=measured, scale=std, size=samples)
        lo, hi = INPUT_BOUNDS[key]
        return np.clip(samples_arr, lo, hi)

    loads  = sample_input(load,  "production_load")
    cycles = sample_input(cycle, "cycle_time")
    temps  = sample_input(temp,  "machine_temperature")
    speeds = sample_input(speed, "axis_speed")

    # ── Step 2: Run model N times with sampled inputs ─────────────────────────
    results = np.empty(samples)

    for i in range(samples):
        pred = model.predict(
            float(loads[i]),
            float(cycles[i]),
            float(temps[i]),
            float(speeds[i])
        )
        # Guard against None (untrained model) or negative predictions
        results[i] = max(float(pred), 0.01) if pred is not None else 0.01

    # ── Step 3: Return distribution + sampled inputs for diagnostics ──────────
    sampled_inputs = {
        "production_load":     loads,
        "cycle_time":          cycles,
        "machine_temperature": temps,
        "axis_speed":          speeds,
    }

    return results, sampled_inputs