"""Best-response function R(m) for the Vance model, Bulmer & Parker (2002)."""
import numpy as np
from .fitness import log_fitness

def _cubic_coefficients(m2, alpha, beta):
    """
    Coefficients of the first-order condition as a cubic in m1,
    """
    return [-1, (alpha + beta - (2*m2)), ((2*alpha*m2) - (m2**2)), (alpha*(m2**2))]

def best_response(m2, alpha, beta):
    """
    Fast version. Find all roots, keep real positive ones, return
    whichever gives the highest log_fitness.
    """
    roots = np.roots(_cubic_coefficients(m2, alpha, beta))
    real = roots[np.abs(roots.imag) < 1e-9].real
    positive = real[real > 0]
    fitness = log_fitness(positive, m2, alpha, beta)
    return positive[np.argmax(fitness)]

def best_response_grid(m2, alpha, beta, hi=None):
    """
    Brute force: evaluate log_fitness on a fine grid, return the argmax.
    Slow but always finds the global maximum.
    """
    if hi is None:
        hi = 2*(alpha+beta)
    m1 = np.linspace(0, hi, 20000)[1:]
    return m1[np.argmax(log_fitness(m1, m2, alpha, beta))]

def best_response_slope(m, alpha, beta, h=1e-5):
    """
    Numerical derivative of best_response at m.
    """
    m1_up = best_response(m + h, alpha, beta)
    m1_down = best_response(m - h, alpha, beta)
    return (m1_up - m1_down) / (2 * h)

