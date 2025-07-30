from __future__ import annotations

import numpy as np


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_pct = e_hist / max(e_hist.sum(), 1) + epsilon
    a_pct = a_hist / max(a_hist.sum(), 1) + epsilon
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def window_psi(reference_df, current_df, features: list[str]) -> dict[str, float]:
    out = {}
    for f in features:
        if f in reference_df.columns and f in current_df.columns:
            out[f] = population_stability_index(
                reference_df[f].to_numpy(),
                current_df[f].to_numpy(),
            )
    return out


def max_psi(reference_df, current_df, features: list[str]) -> float:
    psi_map = window_psi(reference_df, current_df, features)
    if not psi_map:
        return 0.0
    return max(psi_map.values())
