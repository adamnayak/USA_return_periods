import pymannkendall as mk
import pandas as pd
import numpy as np

def random_sampling_mk(var_values, sample_size=10000, num_samples=100):
    """
    Perform Mann-Kendall test on random samples and report the median of results, along with additional statistics.

    Args:
        var_values (numpy.ndarray): The time series data.
        sample_size (int): The size of each random sample.
        num_samples (int): The number of random samples to take.

    Returns:
        dict: Aggregated results including median slope, most frequent trend, median p-value, significance, and additional stats.
    """
    slopes = []
    p_values = []
    trends = []
    taus = []

    n = len(var_values)

    # Perform random sampling
    for i in range(num_samples):
        if n > sample_size:
            sample_indices = np.random.choice(n, size=sample_size, replace=False)
            sample = var_values[sample_indices]
        else:
            sample = var_values  # Use the entire dataset if smaller than sample size

        try:
            result = mk.original_test(sample)
            slopes.append(result.slope)
            p_values.append(result.p)
            trends.append(result.trend)
            taus.append(result.Tau)
        except ZeroDivisionError:
            print(f"Skipping sample {i} due to zero-division")
            continue

    # Aggregate results
    median_slope = np.median(slopes) if slopes else np.nan
    median_p_value = np.median(p_values) if p_values else np.nan
    most_frequent_trend = max(set(trends), key=trends.count) if trends else "no trend"
    median_tau = np.median(taus) if taus else np.nan
    median_effect_size = abs(median_tau) if not np.isnan(median_tau) else np.nan
    significant = median_p_value < 0.05 if not np.isnan(median_p_value) else False

    # Approximate confidence interval for Sen's Slope
    if slopes:
        confidence_interval = (median_slope - 1.96 * np.std(slopes), 
                               median_slope + 1.96 * np.std(slopes))
    else:
        confidence_interval = (np.nan, np.nan)

    dof = sample_size - 1 if sample_size <= n else n - 1  # Degrees of Freedom (approximation)

    return {
        "trend": most_frequent_trend,
        "p-value": median_p_value,
        "slope": median_slope,
        "significant": significant,
        "sample_size": sample_size if sample_size <= n else n,
        "test_statistic": median_tau,
        "effect_size": median_effect_size,
        "confidence_interval_lower": confidence_interval[0],
        "confidence_interval_upper": confidence_interval[1],
        "degrees_of_freedom": dof
    }
