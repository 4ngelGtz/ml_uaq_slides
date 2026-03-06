"""
Semana 4 — Figuras reproducibles (MLE / ERM).

Guarda PNGs en slides/figs/ con nombres:
- week04_underflow_loglik.png
- week04_bernoulli_nll.png
- week04_gaussian_nll_mu.png
- week04_01_vs_logloss.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGDIR = Path(__file__).resolve().parents[1] / "slides" / "figs"


def set_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def underflow_demo() -> None:
    """
    Show how product likelihood underflows while log-likelihood stays finite.
    """
    # Assume constant probability 0.01 repeated n times.
    n = np.arange(1, 401)
    p = 1e-2
    # Use float64; will underflow to 0 for large n.
    L = p**n
    logL = n * math.log(p)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.plot(n, L, color="#4c78a8", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("n (número de datos)")
    ax.set_ylabel(r"$L = \prod_i p(x_i|\theta)$ (escala log)")
    ax.set_title("Likelihood como producto (underflow)")

    ax2 = axes[1]
    ax2.plot(n, logL, color="#f58518", linewidth=2)
    ax2.set_xlabel("n (número de datos)")
    ax2.set_ylabel(r"$\ell = \sum_i \log p(x_i|\theta)$")
    ax2.set_title("Log-likelihood como suma (estable)")

    savefig(fig, "week04_underflow_loglik.png")


def bernoulli_nll_curve() -> None:
    """
    NLL(pi) for a small Bernoulli dataset, with MLE marked.
    """
    x = np.array([1, 0, 1, 1, 0], dtype=float)
    n = x.size
    s = x.sum()
    p_hat = s / n

    eps = 1e-12
    ps = np.linspace(0.001, 0.999, 500)
    ps_clip = np.clip(ps, eps, 1 - eps)
    nll = -(s * np.log(ps_clip) + (n - s) * np.log(1 - ps_clip))
    nll = nll / n  # average NLL (more comparable)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(ps, nll, color="#4c78a8", linewidth=2, label="NLL promedio")
    ax.axvline(p_hat, color="k", linestyle="--", linewidth=2, label=rf"$\hat{{\pi}}=\bar x={p_hat:.2f}$")
    ax.set_xlabel(r"$\pi$")
    ax.set_ylabel("NLL promedio")
    ax.set_title(r"Bernoulli: NLL$(\pi)$ y MLE (Murphy 4.2.3)")
    ax.legend(frameon=False, loc="best")
    savefig(fig, "week04_bernoulli_nll.png")


def gaussian_nll_mu_curve() -> None:
    """
    NLL(mu) for univariate Gaussian with sigma^2 fixed.
    """
    x = np.array([1.0, 2.0, 4.0])
    mu_hat = x.mean()

    # Fix sigma^2 to the MLE (for visualization), with small floor.
    s2_hat = np.mean((x - mu_hat) ** 2)
    s2 = max(s2_hat, 1e-6)

    mus = np.linspace(mu_hat - 2.0, mu_hat + 2.0, 400)
    sse = np.sum((x[None, :] - mus[:, None]) ** 2, axis=1)
    n = x.size
    # NLL up to constant: (n/2) log s2 + (1/(2 s2)) * SSE
    nll = 0.5 * n * math.log(s2) + 0.5 * sse / s2
    nll = nll / n  # average

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(mus, nll, color="#4c78a8", linewidth=2, label=rf"NLL promedio (σ² fijo = {s2_hat:.3f})")
    ax.axvline(mu_hat, color="k", linestyle="--", linewidth=2, label=rf"$\hat{{\mu}}=\bar x={mu_hat:.3f}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("NLL promedio (constantes omitidas)")
    ax.set_title(r"Gaussiana 1D: NLL$(\mu)$ (Murphy 4.2.5)")
    ax.legend(frameon=False, loc="best")
    savefig(fig, "week04_gaussian_nll_mu.png")


def zero_one_vs_logloss() -> None:
    """
    Compare 0-1 loss vs log-loss as a function of predicted probability for a fixed label.
    """
    p = np.linspace(1e-4, 1 - 1e-4, 800)

    # For y=1, predicted prob is p.
    y = 1
    logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    # 0-1 requires a decision threshold (use 0.5).
    yhat = (p >= 0.5).astype(int)
    loss01 = (yhat != y).astype(float)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(p, loss01, color="#4c78a8", linewidth=2, label=r"Pérdida $0$-$1$ (umbral 0.5)")
    ax.plot(p, logloss, color="#f58518", linewidth=2, label=r"Log-loss $-\log p(y|x,\theta)$")
    ax.set_xlabel(r"Probabilidad predicha $p=\Pr(y=1)$")
    ax.set_ylabel("Pérdida")
    ax.set_ylim(-0.05, 6.2)
    ax.set_title(r"Surrogate loss: suave vs $0$-$1$ (Murphy 4.3.1–4.3.2)")
    ax.legend(frameon=False, loc="upper right")
    savefig(fig, "week04_01_vs_logloss.png")


def main() -> None:
    set_mpl_style()
    underflow_demo()
    bernoulli_nll_curve()
    gaussian_nll_mu_curve()
    zero_one_vs_logloss()


if __name__ == "__main__":
    main()

