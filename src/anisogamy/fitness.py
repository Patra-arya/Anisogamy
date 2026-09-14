"""Reproductive fitness from Bulmer & Parker (2002), eq. 2.1 and 2.19."""
import numpy as np
from .survival import vance


def log_fitness(m_self, m_partner, alpha, beta, delta=0.0, 
                g=vance, f=vance):
    """
    Log reproductive fitness of an individual making gametes of size
    m_self, when its partner makes m_partner.
    """
    m_self = np.clip(np.asarray(m_self, dtype=float), 1e-300, None)
    m_partner = np.clip(np.asarray(m_partner, dtype=float), 1e-300, None)
    
    n_gam = -np.log(m_self)
    with np.errstate(divide='ignore'):
        gamete_surv = np.log(g(m_self, scale=alpha, delta=delta))
        zygote_surv = np.log(f(m_self + m_partner, scale=beta, delta=2*delta))

    # M/m_self, with ln M dropped as a constant
    return n_gam + gamete_surv + zygote_surv 



def isogamous_ess_vance(alpha, beta):
    """
    Analytic isogamous ESS for the Vance functions: m* = alpha + beta/4.

    Only valid for the Vance case--it's the closed-form solution of
    eq. 2.5, not a general result.
    """
    return alpha + (beta/4.0)