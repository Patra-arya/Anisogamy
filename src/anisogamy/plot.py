"""Visualization tools for best-response dynamics, cobweb plots, and bifurcation diagrams."""
import os
import numpy as np
import matplotlib.pyplot as plt

from .best_response import best_response
from .dynamics import iterate, classify


def plot_best_response(alpha, beta, ax=None, save_path=None):
    """Fig. 1 top row: R(m) against m, with the 45-degree line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    m_vals = np.linspace(0.01, beta * 1.5, 300)
    r_vals = np.array([best_response(m, alpha, beta) for m in m_vals])

    ax.plot(m_vals, r_vals, 'b-', linewidth=2, label=r'$R(m)$')
    ax.plot(m_vals, m_vals, 'k--', linewidth=1, label=r'$m_1 = m_2$')

    ax.set_xlabel(r'Partner Gamete Size ($m_2$)')
    ax.set_ylabel(r'Best Response ($m_1$)')
    ax.set_title(f'Best Response Map ($\\alpha={alpha}$, $\\beta={beta}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
                dir_name = os.path.dirname(save_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                    
                fig = ax.get_figure()
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
    

    return ax


def plot_cobweb(m0, alpha, beta, n=30, ax=None, save_path=None):
    """The staircase: vertical to the curve, horizontal to the diagonal."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    plot_best_response(alpha, beta, ax=ax)

    traj = iterate(m0, alpha, beta, n=n)

    cobweb_x = []
    cobweb_y = []

    x = m0
    for y in traj[1:]:
        cobweb_x.extend([x, x])
        cobweb_y.extend([x, y])
        cobweb_x.extend([x, y])
        cobweb_y.extend([y, y])
        x = y

    ax.plot(cobweb_x, cobweb_y, 'r-', alpha=0.7, linewidth=1.2, label='Cobweb Trajectory')
    ax.scatter([m0], [m0], color='red', zorder=5, label=f'Start ($m_0={m0}$)')
    ax.legend()
    if save_path is not None:
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            fig = ax.get_figure()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def plot_bifurcation(alpha, betas, ax=None, save_path=None):
    """The pitchfork. Not in either paper."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    beta_arr = np.asarray(betas)
    m_iso = alpha + (beta_arr / 4.0)
    stable_mask = beta_arr <= (4 * alpha)

    ax.plot(beta_arr[stable_mask], m_iso[stable_mask], 'k-', 
            linewidth=1.5, label='isogamy (stable)')
    ax.plot(beta_arr[~stable_mask], m_iso[~stable_mask], 'k--', 
            linewidth=1.5, label='isogamy (unstable)')
    
    betas_ani, sizes_lo, sizes_hi = [], [], []
    for b in beta_arr:
        res = classify(alpha, b)
        if res["regime"] == "anisogamy":
            betas_ani.append(b)
            sizes_lo.append(res["sizes"][0])
            sizes_hi.append(res["sizes"][-1])
    if betas_ani:
        ax.plot(betas_ani, sizes_lo, lw=2.2,
                label="small gamete (sperm)")
        ax.plot(betas_ani, sizes_hi, lw=2.2,
                label="large gamete (ovum)")
        ax.legend()


    ax.axvline(4 * alpha, ls=':', lw=1, color='gray', label=rf'$\beta = 4\alpha$')

    ax.set_xlabel(r'Zygote Survival Parameter ($\beta$)')
    ax.set_ylabel(r'Stable Gamete Size ($m^*$)')
    ax.set_title(f'Bifurcation Diagram ($\\alpha={alpha}$)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        fig = ax.get_figure()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax