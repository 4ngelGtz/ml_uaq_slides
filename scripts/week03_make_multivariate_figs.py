"""
Semana 3 — Figuras reproducibles (MVN, condicionamiento, imputación, GMM).

Guarda PNGs en slides/figs/ con nombres:
- week03_cov_corr.png
- week03_uncorrelated_dependent.png
- week03_simpson.png
- week03_mahalanobis.png
- week03_covariance_geometry.png
- week03_conditioning_2d.png
- week03_imputation.png
- week03_imputation_iris.png
- week03_gmm_2d.png
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.datasets import load_iris
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    load_iris = None
    GaussianMixture = None


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


def cov_corr_heatmaps(rng: np.random.Generator) -> None:
    """Heatmaps of covariance vs correlation for a synthetic Gaussian."""
    mu = np.array([0.0, 0.0, 0.0])
    Sigma = np.array(
        [
            [1.0, 0.8, -0.2],
            [0.8, 2.5, -0.4],
            [-0.2, -0.4, 0.5],
        ]
    )
    X = rng.multivariate_normal(mu, Sigma, size=800)
    S = np.cov(X, rowvar=False)
    D = np.sqrt(np.diag(S))
    R = S / np.outer(D, D)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].grid(False)
    axes[1].grid(False)
    im0 = axes[0].imshow(S, cmap="coolwarm")
    axes[0].set_title(r"Covariance $\hat{\Sigma}$")
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(R, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title(r"Correlation $\hat{R}$")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    savefig(fig, "week03_cov_corr.png")


def uncorrelated_but_dependent(rng: np.random.Generator) -> None:
    """Example where corr=0 but variables are dependent (X ~ U(-1,1), Y=X^2 + noise)."""
    n = 1200
    x = rng.uniform(-1, 1, size=n)
    y = x**2 + 0.08 * rng.standard_normal(size=n)
    corr = np.corrcoef(x, y)[0, 1]

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.scatter(x, y, s=14, alpha=0.55, color="#1f77b4")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")
    ax.set_title(f"No correlación no implica independencia\n(corr ≈ {corr:.3f})")
    savefig(fig, "week03_uncorrelated_dependent.png")


def simpson_berkeley_dummy() -> None:
    """Dummy bars for Simpson (aggregate vs stratified)."""
    # Synthetic rates to illustrate inversion.
    depts = ["A", "B", "C", "D", "E", "F"]
    men = np.array([0.62, 0.63, 0.37, 0.33, 0.28, 0.06])
    women = np.array([0.82, 0.68, 0.34, 0.35, 0.24, 0.07])
    agg_m = 0.44
    agg_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    axes[0].bar(["Hombres", "Mujeres"], [agg_m, agg_w], color=["#4c78a8", "#f58518"])
    axes[0].set_title("Agregado")
    axes[0].set_ylabel("Tasa de admisión (dummy)")
    axes[0].set_ylim(0, 1)

    x = np.arange(len(depts))
    w = 0.38
    axes[1].bar(x - w / 2, men, width=w, label="Hombres", color="#4c78a8")
    axes[1].bar(x + w / 2, women, width=w, label="Mujeres", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(depts)
    axes[1].set_title("Por departamento")
    axes[1].legend(frameon=False, loc="upper right")
    savefig(fig, "week03_simpson.png")


def mahalanobis_ellipse_demo(rng: np.random.Generator) -> None:
    """Show samples and Mahalanobis level sets (dummy)."""
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.5, 1.0], [1.0, 1.2]])
    X = rng.multivariate_normal(mu, Sigma, size=600)

    # Grid for level set contours
    xx, yy = np.meshgrid(np.linspace(-4, 4, 220), np.linspace(-4, 4, 220))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    Sinv = np.linalg.inv(Sigma)
    d2 = np.einsum("ni,ij,nj->n", grid - mu, Sinv, grid - mu).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(4.7, 4.0))
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.35, color="#1f77b4")
    cs = ax.contour(xx, yy, d2, levels=[1.0, 4.0, 9.0], colors="k", linewidths=1.6)
    ax.clabel(cs, inline=True, fontsize=9, fmt={1.0: "d=1", 4.0: "d=2", 9.0: "d=3"})
    ax.set_title("Elipses de Mahalanobis (d=c)")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    savefig(fig, "week03_mahalanobis.png")


def covariance_geometry_demo(rng: np.random.Generator) -> None:
    """Ellipse + eigenvectors to visualize Σ geometry."""
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[2.0, 1.2], [1.2, 1.0]])

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(Sigma)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Sample points
    X = rng.multivariate_normal(mu, Sigma, size=500)

    fig, ax = plt.subplots(figsize=(4.9, 4.0))
    ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.35, color="#1f77b4")

    # Plot eigenvectors scaled by sqrt(eigenvalues)
    for k in range(2):
        v = vecs[:, k]
        scale = math.sqrt(vals[k])
        ax.arrow(
            mu[0],
            mu[1],
            2.2 * scale * v[0],
            2.2 * scale * v[1],
            head_width=0.15,
            head_length=0.2,
            color="#d62728" if k == 0 else "#2ca02c",
            linewidth=2.5,
            length_includes_head=True,
        )

    ax.set_title(r"Geometría de $\Sigma$: autovectores/autovalores")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    savefig(fig, "week03_covariance_geometry.png")


def conditioning_2d_demo(rng: np.random.Generator) -> None:
    """Visual intuition: conditional slice in 2D Gaussian."""
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.8, 1.1], [1.1, 1.2]])
    X = rng.multivariate_normal(mu, Sigma, size=900)

    a = 0.8  # condition x1=a
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.28, color="#1f77b4", label="Muestras")
    ax.axvline(a, color="k", linestyle="--", linewidth=2, label=r"Fijar $x_1=a$")
    ax.set_title("Condicionamiento en 2D: rebanada vertical")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    savefig(fig, "week03_conditioning_2d.png")


def generic_imputation_demo(rng: np.random.Generator) -> None:
    """Toy imputation: hide x2 for some points and impute using a linear conditional mean."""
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.5, 0.9], [0.9, 1.2]])
    X = rng.multivariate_normal(mu, Sigma, size=600)

    n = X.shape[0]
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[: int(0.25 * n)]] = True

    # Conditional mean formula for 2D: E[X2|X1=x1] = mu2 + S21/S11 * (x1-mu1)
    mu1, mu2 = mu
    s11 = Sigma[0, 0]
    s21 = Sigma[1, 0]
    x1 = X[mask, 0]
    x2_true = X[mask, 1]
    x2_imp = mu2 + (s21 / s11) * (x1 - mu1)
    mse = ((x2_true - x2_imp) ** 2).mean()

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(x2_true, x2_imp, s=36, alpha=0.8, color="#1f77b4", edgecolors="black", linewidths=0.5)
    lims = [min(x2_true.min(), x2_imp.min()) - 0.2, max(x2_true.max(), x2_imp.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=2)
    ax.set_xlabel(r"$x_2$ verdadero (oculto)")
    ax.set_ylabel(r"$x_2$ imputado")
    ax.set_title(f"Imputación por condicionamiento (toy)\n(25% oculto, MSE={mse:.3f})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    savefig(fig, "week03_imputation.png")


def iris_imputation_example(rng: np.random.Generator) -> None:
    """Iris: ocultar ancho del pétalo (20%), imputar con media condicional MVN; true vs imputado."""
    if load_iris is None:
        return

    data = load_iris()
    X = data.data  # (150, 4): sepal length, sepal width, petal length, petal width
    idx_miss = 3  # petal width
    n, d = X.shape

    # Ocultar 20% de la variable idx_miss
    mask_miss = np.zeros(n, dtype=bool)
    mask_miss[rng.permutation(n)[: int(0.20 * n)]] = True
    obs = ~mask_miss

    # Estimar mu y Sigma con filas observadas (sin "faltante" en este experimento)
    X_obs = X[obs]
    mu_est = X_obs.mean(axis=0)
    Sigma_est = np.cov(X_obs, rowvar=False)
    Sigma_est += 1e-6 * np.eye(d)  # regularización pequeña

    # Observadas: 0,1,2 ; faltante: 3
    idx_o = np.array([0, 1, 2])
    idx_m = np.array([3])
    mu_o = mu_est[idx_o]
    mu_m = mu_est[idx_m]
    S_oo = Sigma_est[np.ix_(idx_o, idx_o)]
    S_mo = Sigma_est[np.ix_(idx_m, idx_o)]
    S_oo_inv = np.linalg.inv(S_oo)

    x_imputed = X.copy()
    for i in np.where(mask_miss)[0]:
        x_o = X[i, idx_o]
        # E[X_m | X_o] = mu_m + S_mo @ inv(S_oo) @ (x_o - mu_o)
        mu_m_cond = mu_m + S_mo @ S_oo_inv @ (x_o - mu_o)
        x_imputed[i, idx_m] = mu_m_cond

    true_vals = X[mask_miss, idx_miss]
    imp_vals = x_imputed[mask_miss, idx_miss]
    n_ocultos = true_vals.size
    mse = ((true_vals - imp_vals) ** 2).mean()

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(
        true_vals,
        imp_vals,
        s=50,
        alpha=0.85,
        color="#1f77b4",
        edgecolors="black",
        linewidths=0.5,
        label=f"Observaciones con valor oculto (n={n_ocultos})",
    )
    lims = [min(true_vals.min(), imp_vals.min()) - 0.1, max(true_vals.max(), imp_vals.max()) + 0.1]
    ax.plot(lims, lims, "k--", linewidth=2, label="Imputación perfecta (y = x)")
    ax.set_xlabel("Ancho del pétalo — valor verdadero (oculto en el experimento)")
    ax.set_ylabel("Ancho del pétalo — valor imputado")
    ax.set_title(
        f"Imputación por condicionamiento — Iris\n"
        f"Cada punto = una fila con ancho del pétalo oculto (20%, MSE = {mse:.4f})"
    )
    ax.legend(frameon=True, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    savefig(fig, "week03_imputation_iris.png")


def _plot_cov_ellipse(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray, color: str) -> None:
    """Plot a covariance ellipse (1-2 std) for 2D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    # 2-sigma ellipse
    radii = 2.0 * np.sqrt(np.maximum(vals, 1e-12))
    ellipse = (vecs @ (radii[:, None] * circle)).T + mean[None, :]
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=2)


def gmm_2d_demo(rng: np.random.Generator) -> None:
    """Synthetic mixture and (if available) fit a GMM; plot points and ellipses."""
    n1, n2 = 300, 260
    mu1 = np.array([-2.0, -1.0])
    S1 = np.array([[1.1, 0.3], [0.3, 0.6]])
    mu2 = np.array([2.3, 1.8])
    S2 = np.array([[0.7, -0.2], [-0.2, 1.2]])

    X1 = rng.multivariate_normal(mu1, S1, size=n1)
    X2 = rng.multivariate_normal(mu2, S2, size=n2)
    X = np.vstack([X1, X2])

    if GaussianMixture is None:
        labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
        means = np.stack([mu1, mu2], axis=0)
        covs = np.stack([S1, S2], axis=0)
        weights = np.array([n1 / (n1 + n2), n2 / (n1 + n2)])
    else:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gmm.fit(X)
        labels = gmm.predict(X)
        means = gmm.means_
        covs = gmm.covariances_
        weights = gmm.weights_

    colors = ["#4c78a8", "#f58518"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for k in range(2):
        pts = X[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.5, color=colors[k], label=f"comp {k} (π={weights[k]:.2f})")
        _plot_cov_ellipse(ax, means[k], covs[k], color=colors[k])
    ax.set_title("GMM 2D: puntos + elipses por componente (dummy)")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    savefig(fig, "week03_gmm_2d.png")


def main() -> None:
    set_mpl_style()
    rng = np.random.default_rng(0)
    cov_corr_heatmaps(rng)
    uncorrelated_but_dependent(rng)
    simpson_berkeley_dummy()
    mahalanobis_ellipse_demo(rng)
    covariance_geometry_demo(rng)
    conditioning_2d_demo(rng)
    generic_imputation_demo(rng)
    iris_imputation_example(rng)
    gmm_2d_demo(rng)


if __name__ == "__main__":
    main()

