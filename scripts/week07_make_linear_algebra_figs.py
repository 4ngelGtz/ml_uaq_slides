"""
Week 7 — Linear algebra figures for slides (minimal, pedagogical Matplotlib).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)

# Sobrio palette
C_FG = "#1f2937"
C_ACCENT = "#2563eb"
C_RES = "#dc2626"
C_PROJ = "#059669"
C_GRID = "#e5e7eb"


def _style_axes(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=C_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_FG)


def _save_fig(fig, fname: str, *, pad_inches: float = 0.22) -> None:
    """Save for Beamer: include legends/suptitles; avoid clipped PNG edges."""
    out = os.path.join(OUTDIR, fname)
    fig.savefig(
        out,
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    plt.close(fig)
    print("saved:", out)


def fig_projection_line() -> None:
    """Projection of y onto direction u; show residual orthogonal."""
    u = np.array([2.0, 1.0])
    y = np.array([1.5, 2.2])
    u = u / np.linalg.norm(u)
    a = float(np.dot(u, y))
    y_hat = a * u
    r = y - y_hat

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    lim = 2.8
    ax.axhline(0, color=C_GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=C_GRID, linewidth=1, zorder=0)

    t = np.linspace(-lim, lim, 200)
    ax.plot(t * u[0], t * u[1], color=C_FG, linewidth=1.8, label="span(u)")

    ax.arrow(0, 0, 2.2 * u[0], 2.2 * u[1], head_width=0.12, head_length=0.12, fc=C_FG, ec=C_FG, length_includes_head=True)
    ax.text(u[0] * 2.35, u[1] * 2.35, r"$u$", fontsize=11, color=C_FG)

    ax.plot([0, y[0]], [0, y[1]], color=C_ACCENT, linewidth=2.0, label=r"$y$")
    ax.plot([0, y_hat[0]], [0, y_hat[1]], color=C_PROJ, linewidth=2.0, linestyle="--", label=r"$\hat y=\mathrm{proj}_u(y)$")
    ax.plot([y_hat[0], y[0]], [y_hat[1], y[1]], color=C_RES, linewidth=2.0, label=r"$r=y-\hat y$")

    ax.scatter([0], [0], s=36, c=C_FG, zorder=5)
    ax.scatter([y[0]], [y[1]], s=40, c=C_ACCENT, zorder=5, edgecolors="white", linewidths=0.8)
    ax.scatter([y_hat[0]], [y_hat[1]], s=40, c=C_PROJ, zorder=5, edgecolors="white", linewidths=0.8)

    # Margins: legend (upper left) and long math labels need space; avoid clip in Beamer.
    ax.set_xlim(-1.15, lim + 0.35)
    ax.set_ylim(-0.55, lim + 0.45)
    ax.set_title("Proyeccion ortogonal sobre una recta", fontsize=11, color=C_FG, pad=12)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92, borderpad=0.6)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, "week07_projection_line.png")


def fig_ls_scatter() -> None:
    """Worked example: design matrix with intercept + one feature, LS line."""
    x_feat = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 2.0])
    w0, w1 = 7.0 / 6.0, 0.5
    xs = np.linspace(-0.2, 2.3, 100)
    ys = w0 + w1 * xs

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(x_feat, y, s=55, c=C_ACCENT, zorder=5, edgecolors="white", linewidths=0.9, label="datos $(x_i,y_i)$")
    ax.plot(xs, ys, color=C_PROJ, linewidth=2.2, label=r"recta LS: $\hat y = \hat w_0 + \hat w_1 x$")

    y_hat = w0 + w1 * x_feat
    for xi, yi, yhi in zip(x_feat, y, y_hat):
        ax.plot([xi, xi], [yi, yhi], color=C_RES, linewidth=1.5, alpha=0.85)

    ax.set_xlabel(r"feature $x$", fontsize=10, color=C_FG)
    ax.set_ylabel(r"objetivo $y$", fontsize=10, color=C_FG)
    ax.set_title("Minimos cuadrados: ajuste y residuos verticales", fontsize=11, color=C_FG, pad=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.55, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_FG)
    ax.margins(x=0.12, y=0.14)
    fig.tight_layout()
    _save_fig(fig, "week07_ls_scatter.png")


def fig_column_space_3d() -> None:
    """y projected onto z=0 plane (column space of [e1 e2]); residual along z."""
    y = np.array([1.2, 0.7, 1.6])
    y_hat = np.array([y[0], y[1], 0.0])
    r = y - y_hat

    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(111, projection="3d")

    s = 1.6
    xx, yy = np.meshgrid(np.linspace(-0.2, s, 8), np.linspace(-0.2, s, 8))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.22, color=C_ACCENT, shade=False)

    ax.plot([0, y[0]], [0, y[1]], [0, y[2]], color=C_ACCENT, linewidth=2.4, label=r"$y$")
    ax.plot([0, y_hat[0]], [0, y_hat[1]], [0, y_hat[2]], color=C_PROJ, linewidth=2.2, linestyle="--", label=r"$\hat y$ (en $\mathrm{col}(X)$)")
    ax.plot([y_hat[0], y[0]], [y_hat[1], y[1]], [y_hat[2], y[2]], color=C_RES, linewidth=2.0, label=r"$r \perp \mathrm{col}(X)$")

    ax.scatter([0], [0], [0], s=28, c=C_FG)
    ax.set_xlim(0, s)
    ax.set_ylim(0, s)
    ax.set_zlim(0, s * 1.1)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    ax.set_zlabel("z", fontsize=9)
    ax.set_title(r"Proyeccion de $y$ sobre un plano (espacio columna)", fontsize=10, pad=14)
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(-0.02, 1.02), framealpha=0.95)
    ax.view_init(elev=22, azim=-55)
    ax.grid(False)
    fig.tight_layout()
    _save_fig(fig, "week07_column_space_3d.png", pad_inches=0.28)


def fig_circle_ellipse(A: np.ndarray, title: str, fname: str) -> None:
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    ellipse = A @ circle

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, pts, ttl in zip(
        axes,
        [circle, ellipse],
        [r"Circulo unitario $\|x\|_2=1$", r"Imagen $Ax$ (elipse)"],
    ):
        ax.plot(pts[0], pts[1], color=C_ACCENT, linewidth=2.0)
        ax.axhline(0, color=C_GRID, linewidth=1)
        ax.axvline(0, color=C_GRID, linewidth=1)
        ax.set_title(ttl, fontsize=10, color=C_FG)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        lim = max(2.8, float(np.max(np.abs(pts))) * 1.22)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        for spine in ax.spines.values():
            spine.set_color(C_FG)

    fig.suptitle(title, fontsize=11, color=C_FG, y=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])
    _save_fig(fig, fname, pad_inches=0.24)


def fig_sym2x2_quadratic_ellipse() -> None:
    """Concrete symmetric 2x2 example: unit circle mapped to ellipse."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle

    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(circle[0], circle[1], linestyle="--", linewidth=1.4, color=C_FG, alpha=0.85, label=r"$\|v\|_2=1$")
    ax.plot(ellipse[0], ellipse[1], linewidth=2.2, color=C_ACCENT, label=r"$Av,\ A=[[2,1],[1,2]]$")

    # Principal axes of the ellipse: lengths are eigenvalues for SPD A.
    for lam, vec, col in zip(eigvals, eigvecs.T, [C_PROJ, C_RES]):
        p = lam * vec
        ax.plot([-p[0], p[0]], [-p[1], p[1]], color=col, linewidth=1.6)

    ax.text(3.1, 3.15, r"$\lambda_1=3$", fontsize=9, color=C_PROJ)
    ax.text(0.95, -1.05, r"$\lambda_2=1$", fontsize=9, color=C_RES)
    ax.set_title(r"Forma cuadratica simetrica: circulo $\rightarrow$ elipse", fontsize=10, color=C_FG, pad=10)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.55, color=C_GRID, zorder=0)
    ax.axhline(0, color=C_GRID, linewidth=1, zorder=1)
    ax.axvline(0, color=C_GRID, linewidth=1, zorder=1)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.2, 3.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    _style_axes(ax)
    fig.tight_layout()
    _save_fig(fig, "week07_sym2x2_ellipse.png", pad_inches=0.26)


def fig_svd_scree() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 8))
    s = np.linalg.svd(A, compute_uv=False)
    s = np.sort(s)[::-1]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(np.arange(1, len(s) + 1), s, color=C_ACCENT, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("indice $i$", fontsize=10, color=C_FG)
    ax.set_ylabel(r"valor singular $\sigma_i$", fontsize=10, color=C_FG)
    ax.set_title("Valores singulares (scree plot)", fontsize=11, color=C_FG)
    ax.tick_params(colors=C_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_FG)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55, color=C_GRID)
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    _save_fig(fig, "week07_svd_scree.png")


def fig_svd_trunc_reconstruction() -> None:
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 5))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    ranks = [1, 2, A.shape[1]]
    errors = []
    for k in ranks:
        Sk = np.zeros_like(s)
        Sk[:k] = s[:k]
        Ak = (U * Sk) @ Vt
        errors.append(np.linalg.norm(A - Ak, "fro"))

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.bar([str(k) for k in ranks], errors, color=C_PROJ, edgecolor="white", linewidth=0.6)
    ax.set_xlabel(r"rango truncado $k$", fontsize=10, color=C_FG)
    ax.set_ylabel(r"$\|A-A_k\|_F$", fontsize=10, color=C_FG)
    ax.set_title("SVD truncada: error de reconstruccion (Frobenius)", fontsize=10, color=C_FG, pad=10)
    ax.tick_params(colors=C_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_FG)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55, color=C_GRID)
    ax.margins(x=0.06, y=0.12)
    fig.tight_layout()
    _save_fig(fig, "week07_svd_trunc.png")


def fig_pca_2d() -> None:
    rng = np.random.default_rng(2)
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.6, 1.1], [1.1, 1.0]])
    X = rng.multivariate_normal(mean, cov, size=200)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v1 = Vt[0]
    v2 = Vt[1]
    scale = 2.8

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(Xc[:, 0], Xc[:, 1], s=18, alpha=0.55, c=C_FG, edgecolors="none")
    ax.arrow(0, 0, scale * v1[0], scale * v1[1], width=0.04, head_width=0.18, fc=C_ACCENT, ec=C_ACCENT, length_includes_head=True, zorder=6)
    ax.arrow(0, 0, scale * 0.55 * v2[0], scale * 0.55 * v2[1], width=0.035, head_width=0.15, fc=C_PROJ, ec=C_PROJ, length_includes_head=True, zorder=5)
    ax.text(scale * v1[0] * 1.08, scale * v1[1] * 1.08, r"PC1 ($v_1$)", fontsize=9, color=C_ACCENT)
    ax.text(scale * 0.55 * v2[0] * 1.15, scale * 0.55 * v2[1] * 1.15, r"PC2", fontsize=9, color=C_PROJ)
    ax.set_title("PCA 2D: direcciones principales (SVD de $X_c$)", fontsize=10, color=C_FG, pad=10)
    ax.set_xlabel(r"feature 1 (centrada)", fontsize=9, color=C_FG)
    ax.set_ylabel(r"feature 2 (centrada)", fontsize=9, color=C_FG)
    ax.axhline(0, color=C_GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=C_GRID, linewidth=1, zorder=0)
    _style_axes(ax)
    # Room for arrow heads and PC labels outside point cloud
    tip_x = np.array(
        [0.0, scale * v1[0], scale * 0.55 * v2[0], np.min(Xc[:, 0]), np.max(Xc[:, 0])]
    )
    tip_y = np.array(
        [0.0, scale * v1[1], scale * 0.55 * v2[1], np.min(Xc[:, 1]), np.max(Xc[:, 1])]
    )
    pad = 0.55
    ax.set_xlim(tip_x.min() - pad, tip_x.max() + pad)
    ax.set_ylim(tip_y.min() - pad, tip_y.max() + pad)
    fig.tight_layout()
    _save_fig(fig, "week07_pca_2d.png")


def fig_conditioning_compare() -> None:
    A_good = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    A_bad = np.array([[1.0, 1.0], [1.0, 1.0001]], dtype=float)

    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    titles = [
        r"Bien condicionada: $\kappa \approx 1$",
        r"Casi colineal: $\kappa \gg 1$",
    ]
    for ax, A, ttl in zip(axes, [A_good, A_bad], titles):
        pts = A @ circle
        ax.plot(pts[0], pts[1], color=C_ACCENT, linewidth=2.0)
        ax.axhline(0, color=C_GRID, linewidth=1)
        ax.axvline(0, color=C_GRID, linewidth=1)
        ax.set_title(ttl, fontsize=10, color=C_FG)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        lim = max(1.5, float(np.max(np.abs(pts))) * 1.18)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        for spine in ax.spines.values():
            spine.set_color(C_FG)

    fig.suptitle(r"Imagen del circulo unitario: $Ax$", fontsize=11, color=C_FG, y=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])
    _save_fig(fig, "week07_conditioning_ellipses.png", pad_inches=0.24)


def fig_svd_eigenvalues_bars() -> None:
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 4))
    s = np.linalg.svd(A, compute_uv=False)
    w = np.linalg.eigvalsh(A.T @ A)
    w = np.sort(w)[::-1]
    sig2 = s**2

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    x = np.arange(1, len(sig2) + 1)
    w = w[: len(sig2)]
    w = np.maximum(w, 0.0)
    ax.bar(x - 0.2, sig2, width=0.35, label=r"$\sigma_i^2$", color=C_ACCENT, edgecolor="white", linewidth=0.5)
    ax.bar(x + 0.2, w, width=0.35, label=r"$\lambda_i(A^\top A)$", color=C_PROJ, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"indice $i$", fontsize=10, color=C_FG)
    ax.set_ylabel("valor", fontsize=10, color=C_FG)
    ax.set_title(r"SVD y EVD: $\sigma_i^2 = \lambda_i(A^\top A)$", fontsize=10, color=C_FG, pad=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.tick_params(colors=C_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_FG)
    ax.grid(True, axis="y", linestyle=":", alpha=0.55, color=C_GRID)
    ax.margins(x=0.06, y=0.14)
    fig.tight_layout()
    _save_fig(fig, "week07_svd_vs_ata_eigen.png", pad_inches=0.28)


def main() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig_projection_line()
    fig_ls_scatter()
    fig_column_space_3d()
    A_sym = np.array([[2.0, 0.6], [0.6, 1.0]])
    fig_circle_ellipse(
        A_sym,
        r"Matriz simetrica: circulo $\rightarrow$ elipse (ejes principales)",
        "week07_circle_ellipse_symmetric.png",
    )
    fig_sym2x2_quadratic_ellipse()
    rng = np.random.default_rng(4)
    A_gen = rng.normal(size=(2, 2))
    fig_circle_ellipse(
        A_gen,
        r"Matriz general $A$: rotar--estirar--rotar (SVD)",
        "week07_circle_ellipse_general.png",
    )
    fig_svd_scree()
    fig_svd_trunc_reconstruction()
    fig_pca_2d()
    fig_conditioning_compare()
    fig_svd_eigenvalues_bars()


if __name__ == "__main__":
    main()
