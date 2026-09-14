"""Iteration of the best-response map, and the isogamy/anisogamy transition."""
import numpy as np
from .best_response import best_response


def iterate(m0, alpha, beta, n=200):
    """
    Trajectory of alternating best responses, starting from m0.

    Each step is one mating type optimising against the other's current
    strategy. Returns an array of length n+1.
    """
    traj = [m0]
    for _ in range(n):
        traj.append(best_response(traj[-1], alpha, beta))
    return np.array(traj)


def classify(alpha, beta, m0=None, n=500, tol=1e-6):
    """
    Run the iteration to its attractor and report what it found.

    Returns a dict with keys 'regime' ('isogamy' or 'anisogamy') and
    'sizes' (one value or two).

    Method: take the last handful of iterates. If they're all the same,
    it converged to a point. If they alternate between two values, it's
    a 2-cycle.
    """
    if m0 is None:
        # Default perturbation away from isogamous ESS to detect instability
        m0 = (alpha + (beta / 4.0)) * 1.05

    traj = iterate(m0, alpha, beta, n=n)
    
    # Inspect the final states of the trajectory
    last_third = traj[-3]
    last_even = traj[-2]
    last_odd = traj[-1]

    if (np.isclose(last_even, last_odd, atol=tol, rtol=tol) & 
        np.isclose(last_third, last_odd, atol=tol, rtol=tol)):
        return {
            "regime": "isogamy",
            "sizes": np.array([last_odd])
        }
    else:
        return {
            "regime": "anisogamy",
            "sizes": np.array([min(last_even, last_odd), max(last_even, last_odd)])
        }


def bifurcation_sweep(alpha, betas, **kwargs):
    """
    Call classify across a range of beta. Returns arrays suitable
    for plotting stable size(s) against beta.
    """
    beta_list = []
    size_list = []

    for beta in betas:
        result = classify(alpha, beta, **kwargs)
        for s in result["sizes"]:
            beta_list.append(beta)
            size_list.append(s)

    return np.array(beta_list), np.array(size_list)