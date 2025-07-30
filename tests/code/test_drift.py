import numpy as np
import pandas as pd

from modelgate.drift import max_psi, population_stability_index


def test_psi_zero_on_identical():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert population_stability_index(x, x.copy()) < 0.01


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    small = population_stability_index(ref, rng.normal(0.1, 1, 5000))
    big = population_stability_index(ref, rng.normal(1.0, 1, 5000))
    assert big > small
    assert big > 0.2


def test_max_psi_picks_largest_per_feature():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(5, 2, 5000)})
    cur = pd.DataFrame({"a": rng.normal(2, 1, 5000), "b": rng.normal(5, 2, 5000)})
    out = max_psi(ref, cur, features=["a", "b"])
    assert out > 0.2
