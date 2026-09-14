# tests/test_survival.py
import numpy as np
from anisogamy.survival import (
    vance, vance_derivative, complementary_exponential, threshold
)

def test_vance_limits():
    """Zero at the origin, approaching 1 for large x."""
    assert np.isclose(vance(1e-9, scale=2.0), 0.0, atol=1e-5)
    np.testing.assert_allclose(vance(1e6, scale=2.0), 1.0, rtol=1e-5)

def test_vance_inflection():
    """Second derivative crosses zero at x = scale/2."""
    scale = 2.0
    x = np.linspace(0.05, 3 * scale, 20000)
    d2 = np.gradient(np.gradient(vance(x, scale), x), x)
    crossing = x[np.where(np.diff(np.sign(d2)))[0][0]]  # find where sign(d2) changes
    np.testing.assert_allclose(crossing, scale / 2, rtol=1e-2)

def test_vance_derivative_matches_numeric():
    """Analytic derivative agrees with a central difference."""
    scale = 2.0
    x = np.array([0.5, 1.0, 2.0, 4.0])
    h = 1e-5
    num_der = (vance(x + h, scale) - vance(x - h, scale)) / (2 * h)
    ana_der = vance_derivative(x, scale)
    np.testing.assert_allclose(ana_der, num_der, rtol=1e-5)

def test_complementary_exponential_is_concave():
    """Second derivative negative everywhere above delta - never accelerates."""
    scale = 2.0
    x = np.linspace(0.1, 10.0, 500)
    h = 1e-5
    d2 = (complementary_exponential(x + h, scale) 
          - 2 * complementary_exponential(x, scale) 
          + complementary_exponential(x - h, scale)) / (h**2)
    assert np.all(d2 < 0)

def test_all_zero_below_delta():
    """All three families return exactly 0 at and below delta."""
    scale = 2.0
    delta = 0.0
    
    x = np.linspace(-2.0, delta, 1000)
    
    v = vance(x, scale, delta=delta)
    v_d = vance_derivative(x, scale, delta=delta)
    c_e = complementary_exponential(x, scale, delta=delta)
    assert np.all(v == 0.0)
    assert np.all(v_d == 0.0)
    assert np.all(c_e == 0.0)