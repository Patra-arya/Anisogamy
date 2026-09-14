import numpy as np
import pytest
from anisogamy.fitness import log_fitness, isogamous_ess_vance


def test_isogamous_ess_known_values():
    """
    From the Fig. 1 captions: (alpha=1, beta=1) -> 1.25, (1, 10) -> 3.5.
    """
    m_star_1 = isogamous_ess_vance(alpha=1.0, beta=1.0)
    m_star_2 = isogamous_ess_vance(alpha=1.0, beta=10.0)

    assert np.isclose(m_star_1, 1.25)
    assert np.isclose(m_star_2, 3.5)


def test_gradient_vanishes_at_ess():
    """
    At m1 = m2 = m*, the derivative of log_fitness w.r.t. m1 is zero.
    """
    alpha, beta = 2.0, 3.0
    m_star = isogamous_ess_vance(alpha, beta)
    
    # Hold m_partner fixed at m* and test derivative at m_self = m*
    m_self = m_star
    m_partner = m_star
    h = 1e-6

    nudge_up = log_fitness(m_self + h, m_partner, alpha, beta)
    nudge_down = log_fitness(m_self - h, m_partner, alpha, beta)
    d1 = (nudge_up - nudge_down)/(2*h)

    np.testing.assert_allclose(d1, 0.0, atol=1e-6)
