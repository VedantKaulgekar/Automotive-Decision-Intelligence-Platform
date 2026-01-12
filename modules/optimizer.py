import numpy as np
from itertools import product
# -------------------------------------------------------------
# FAST RANDOM OPTIMIZER (100x faster than grid search)
# -------------------------------------------------------------

# -------------------------------------------------------------
# SAFE FAST OPTIMIZER (handles string labels too)
# -------------------------------------------------------------
def _convert_output(val):
    """Convert classifier labels to numeric scores automatically."""
    if isinstance(val, str):
        mapping = {"Low": 0, "Medium": 1, "High": 2}
        return mapping.get(val, 0)
    return float(val)

def optimize_with_model(model, objective="min", n_samples=1500):
    loads     = np.random.uniform(0.3, 1.0, n_samples)
    cycles    = np.random.uniform(20, 80, n_samples)
    temps     = np.random.uniform(30, 120, n_samples)
    speeds    = np.random.uniform(0.2, 3.0, n_samples)

    best_value = None
    best_params = None

    for L, C, T, S in zip(loads, cycles, temps, speeds):
        try:
            raw_val = model.predict(L, C, T, S)
            val = _convert_output(raw_val)   # << 🔥 FIX HERE
        except:
            continue

        cond = (
            best_value is None or 
            (val < best_value if objective == "min" else val > best_value)
        )

        if cond:
            best_value = val
            best_params = {
                "load": float(L),
                "cycle": float(C),
                "temp": float(T),
                "speed": float(S),
                "score": float(val),
                "raw_prediction": raw_val  # keep original class label
            }

    return best_params




# -------------------------------------------------------------
# MULTI-OBJECTIVE: ENERGY + EMISSION + COST + MAINTENANCE
# Creates Pareto front
# -------------------------------------------------------------
def pareto_optimize(models, n_samples=2000):
    energy_model, eff_model, emis_model, maint_model = models

    loads  = np.random.uniform(0.3, 1.0, n_samples)
    cycles = np.random.uniform(20, 80, n_samples)
    temps  = np.random.uniform(30, 120, n_samples)
    speeds = np.random.uniform(0.2, 3.0, n_samples)

    population = []

    for L, C, T, S in zip(loads, cycles, temps, speeds):
        try:
            E  = energy_model.predict(L, C, T, S)
            Ef = eff_model.predict(L, C, T, S)
            Em = emis_model.predict(L, C, T, S)
            M  = maint_model.predict(L, C, T, S)
        except:
            continue

        population.append([L, C, T, S, E, Em, M])

    pop = np.array(population)

    if len(pop) == 0:
        return []

    metrics = pop[:, 4:7] 
    pareto_idx = []

    for i in range(len(metrics)):
        dominated = False
        for j in range(len(metrics)):
            if all(metrics[j] <= metrics[i]) and any(metrics[j] < metrics[i]):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    return pop[pareto_idx].tolist()
