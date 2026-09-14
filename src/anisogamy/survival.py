# Updated implementation for src/anisogamy/survival.py
import numpy as np

def vance(x, scale, delta=0.0):
    x = np.asarray(x, dtype=float)
    shift = x - delta
    val = np.exp(-scale / np.maximum(shift, 1e-10))
    return np.where(shift > 0, val, 0.0)

def vance_derivative(x, scale, delta=0.0):
    x = np.asarray(x, dtype=float)
    shift = x - delta
    der = (scale / np.maximum(shift, 1e-10)**2) * np.exp(-scale / np.maximum(shift, 1e-10))
    return np.where(shift > 0, der, 0.0)

def complementary_exponential(x, scale, delta=0.0):
    x = np.asarray(x, dtype=float)
    shift = x - delta
    val = 1.0 - np.exp(-shift / scale)
    return np.where(shift > 0, val, 0.0)

def threshold(x, scale=None, delta=0.0):
    x = np.asarray(x, dtype=float)
    return np.where(x >= delta, 1.0, 0.0)
