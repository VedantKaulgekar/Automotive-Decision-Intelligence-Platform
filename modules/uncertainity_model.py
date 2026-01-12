import numpy as np

def monte_carlo_energy(energy_value, samples=200):
    # Prevent negative or zero std deviations
    energy_value = max(energy_value, 0.01)

    std = max(0.1 * energy_value, 0.01)

    noise = np.random.normal(0, std, samples)

    distribution = energy_value + noise

    # Ensure no negative simulated values
    distribution[distribution < 0] = 0.01

    return distribution

