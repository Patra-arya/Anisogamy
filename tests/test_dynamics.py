import numpy as np
from anisogamy.dynamics import classify
import pytest

def test_isogamy_below_threshold():
    """beta = 1, alpha = 1 -> isogamy at m* = 1.25."""
    res = classify(alpha=1.0, beta=1.0)
    assert res['regime'] == 'isogamy'
    assert np.isclose(res['sizes'][0], 1.25)


def test_anisogamy_matches_fig1f():
    """beta = 10, alpha = 1 -> (1.13, 8.87)."""
    res = classify(alpha=1.0, beta=10.0)
    assert res['regime'] == 'anisogamy'
    np.testing.assert_allclose(res['sizes'], np.array([1.13, 8.87]), atol=1e-2)

def test_zygote_size_is_smith_fretwell():
    """In the anisogamous regime, the two sizes sum to beta."""
    alpha, beta = 1.0, 10.0
    res = classify(alpha=alpha, beta=beta)
    assert res['regime'] == 'anisogamy'
    # Zygote size m1 + m2 equals beta in the strong anisogamy limit
    np.testing.assert_allclose(np.sum(res['sizes']), beta, atol=1e-1)

def test_transition_at_four_alpha():
    res_1 = classify(alpha=1.0, beta=3.5)
    res_2 = classify(alpha=1.0, beta=6.0)
    assert res_1['regime'] == 'isogamy'
    assert res_2['regime'] == 'anisogamy'