import numpy as np
import pytest
from anisogamy.best_response import (
    best_response, best_response_grid, best_response_slope
)
from anisogamy.fitness import isogamous_ess_vance


def test_best_response_fixes_isogamous_ess():
    """At the isogamous ESS, the best reply to m* is m* itself.

    This is what 'equilibrium' means: R(m*) = m*.
    """
    for alpha, beta in [(1.0, 1.0), (1.0, 10.0), (2.0, 3.0)]:
        m_star = isogamous_ess_vance(alpha, beta)
        np.testing.assert_allclose(best_response(m_star, alpha, beta),
                                   m_star, rtol=1e-8)


def test_best_response_slope_matches_analytic():
    """R'(m*) = -beta / (4*alpha)  (eq. 2.9)."""
    for alpha, beta in [(1.0, 1.0), (1.0, 4.0), (1.0, 10.0), (2.0, 3.0)]:
        m_star = isogamous_ess_vance(alpha, beta)
        expected = -beta / (4.0 * alpha)
        np.testing.assert_allclose(
            best_response_slope(m_star, alpha, beta), expected, rtol=1e-4
        )


def test_cubic_and_grid_agree():
    """The fast cubic solver matches brute force across a beta range
    spanning both the 4*alpha and 32*alpha thresholds."""
    alpha = 1.0
    for beta in [1.0, 4.0, 10.0, 20.0, 32.0, 40.0]:
        for m2 in [0.5, 2.0, 5.0, 11.0]:
            fast = best_response(m2, alpha, beta)
            slow = best_response_grid(m2, alpha, beta)
            np.testing.assert_allclose(fast, slow, rtol=1e-3)