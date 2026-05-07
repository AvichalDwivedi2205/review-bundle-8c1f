"""Statistical helpers for the paper figures."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_mean_ci(values: list[float], samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Return a 95% bootstrap confidence interval for the mean."""
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    array = np.array(values, dtype=float)
    means = []
    for _ in range(samples):
        resample = rng.choice(array, size=len(array), replace=True)
        means.append(float(resample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_wilcoxon(left: list[float], right: list[float]) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test wrapper."""
    if len(left) != len(right) or not left:
        return {"statistic": 0.0, "pvalue": 1.0}
    statistic, pvalue = wilcoxon(left, right)
    return {"statistic": float(statistic), "pvalue": float(pvalue)}

