"""
Week 9 — Linear regression & regularization figures (pedagogical Matplotlib).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)

C_FG = "#1f2937"
C_ACCENT = "#2563eb"
C_RES = "#dc2626"
C_PROJ = "#059669"
C_GRID = "#e5e7eb"
C_WARN = "#d97706"


def _save_fig(fig, fname: str, *, pad_inches: float = 0.22) -> None:
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight",
                pad_inches=pad_inches)
    plt.close(fig)
    print("saved:", out)


# ------------------------------------------------------------------
# 1) OLS line + residuals
# ------------------------------------------------------------------
def fig_ols_line_and_residuals() -> None:
    rng = np.random.default_rng(42)
    n = 30
    x = rng.uniform(0, 10, n)
    y_true = 2.0 + 1.5 * x
    y = y_true + rng.normal(0, 2.5, n)

    X = np.column_stack([np.ones(n), x])
    w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ w_hat

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(x, y, s=30, c=C_ACCENT, edgecolors="white", linewidths=0.6,
               zorder=5, label="datos")
    order = np.argsort(x)
    ax.plot(x[order], y_hat[order], color=C_PROJ, linewidth=2.2,
            label=rf"recta OLS: $\hat y = {w_hat[0]:.1f} + {w_hat[1]:.2f}\,x$")

    for xi, yi, yhi in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, yhi], color=C_RES, linewidth=0.8, alpha=0.6)

    ax.set_xlabel("feature $x$", fontsize=10, color=C_FG)
    ax.set_ylabel("target $y$", fontsize=10, color=C_FG)
    ax.set_title("OLS: recta ajustada y residuos", fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_ols_line_and_residuals.png")


# ------------------------------------------------------------------
# 2) GD loss curve
# ------------------------------------------------------------------
def fig_gd_loss_curve() -> None:
    rng = np.random.default_rng(42)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 2.5, n)
    Xt = np.column_stack([np.ones(n), x])
    theta = np.zeros(2)
    eta = 0.003
    losses = []
    for _ in range(300):
        r = Xt @ theta - y
        losses.append(float(r @ r / n))
        theta -= (2.0 / n) * eta * (Xt.T @ r)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(losses, color=C_ACCENT, linewidth=2.0)
    ax.set_xlabel("iteracion", fontsize=10, color=C_FG)
    ax.set_ylabel("MSE", fontsize=10, color=C_FG)
    ax.set_title("Gradient descent: perdida vs iteracion", fontsize=11,
                 color=C_FG, pad=10)
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_gd_loss_curve.png")


# ------------------------------------------------------------------
# 3) Learning rate comparison
# ------------------------------------------------------------------
def fig_gd_lr_comparison() -> None:
    rng = np.random.default_rng(42)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 2.0 + 1.5 * x + rng.normal(0, 2.5, n)
    Xt = np.column_stack([np.ones(n), x])
    iters = 200
    etas = [0.001, 0.003, 0.012]
    colors = [C_PROJ, C_ACCENT, C_RES]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for eta, c in zip(etas, colors):
        theta = np.zeros(2)
        losses = []
        for _ in range(iters):
            r = Xt @ theta - y
            losses.append(float(r @ r / n))
            theta -= (2.0 / n) * eta * (Xt.T @ r)
        ax.plot(losses, color=c, linewidth=1.8, label=rf"$\eta={eta}$")

    ax.set_xlabel("iteracion", fontsize=10, color=C_FG)
    ax.set_ylabel("MSE", fontsize=10, color=C_FG)
    ax.set_title("Efecto de la tasa de aprendizaje", fontsize=11, color=C_FG,
                 pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_gd_learning_rate_comparison.png")


# ------------------------------------------------------------------
# 4) Ridge vs OLS coefficients (multicollinearity)
# ------------------------------------------------------------------
def fig_ridge_vs_ols_coefs() -> None:
    rng = np.random.default_rng(7)
    n = 50
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.05, n)
    y = 3.0 * x1 + 2.0 * x2 + rng.normal(0, 1.0, n)

    X = np.column_stack([x1, x2])
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    w_ols = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    lam = 5.0
    w_ridge = np.linalg.solve(Xc.T @ Xc + lam * np.eye(2), Xc.T @ yc)

    labels = ["$w_1$", "$w_2$"]
    x_pos = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x_pos - width / 2, w_ols, width, label="OLS", color=C_ACCENT,
           edgecolor="white", linewidth=0.6)
    ax.bar(x_pos + width / 2, w_ridge, width, label=rf"Ridge ($\lambda={lam}$)",
           color=C_PROJ, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=C_FG, linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("valor del coeficiente", fontsize=10, color=C_FG)
    ax.set_title("OLS vs Ridge (features correlacionadas)", fontsize=11,
                 color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_ridge_vs_ols_coefficients.png")


# ------------------------------------------------------------------
# 5) Ridge regularization path
# ------------------------------------------------------------------
def fig_ridge_path() -> None:
    rng = np.random.default_rng(9)
    n, d = 50, 6
    X = rng.normal(0, 1, (n, d))
    w_true = np.array([3.0, -2.0, 0.0, 1.5, 0.0, -0.5])
    y = X @ w_true + rng.normal(0, 1.0, n)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    lambdas = np.logspace(-2, 4, 120)
    coefs = []
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    for lam in lambdas:
        w = np.linalg.solve(XtX + lam * np.eye(d), Xty)
        coefs.append(w)
    coefs = np.array(coefs)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for j in range(d):
        ax.plot(lambdas, coefs[:, j], linewidth=1.6,
                label=rf"$w_{j+1}$")
    ax.set_xscale("log")
    ax.axhline(0, color=C_FG, linewidth=0.6, linestyle="--")
    ax.set_xlabel(r"$\lambda$", fontsize=10, color=C_FG)
    ax.set_ylabel("coeficiente", fontsize=10, color=C_FG)
    ax.set_title("Ridge: trayectoria de coeficientes", fontsize=11,
                 color=C_FG, pad=10)
    ax.legend(fontsize=7, ncol=3, framealpha=0.95, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_ridge_path.png")


# ------------------------------------------------------------------
# 6) Lasso sparse coefficients (OLS vs Ridge vs Lasso)
# ------------------------------------------------------------------
def fig_lasso_sparse_coefs() -> None:
    rng = np.random.default_rng(11)
    n, d = 80, 8
    X = rng.normal(0, 1, (n, d))
    w_true = np.array([4.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ w_true + rng.normal(0, 1.5, n)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    w_ols = np.linalg.lstsq(Xc, yc, rcond=None)[0]

    lam_r = 2.0
    w_ridge = np.linalg.solve(Xc.T @ Xc + lam_r * np.eye(d), Xc.T @ yc)

    lam_l = 3.0
    w_lasso = np.zeros(d)
    eta_l = 0.002
    for _ in range(4000):
        r = Xc @ w_lasso - yc
        g = (2.0 / n) * (Xc.T @ r) + lam_l * np.sign(w_lasso)
        w_lasso -= eta_l * g

    labels = [f"$w_{j+1}$" for j in range(d)]
    x_pos = np.arange(d)
    w_bar = 0.25

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x_pos - w_bar, w_ols, w_bar, label="OLS", color=C_ACCENT,
           edgecolor="white", linewidth=0.5)
    ax.bar(x_pos, w_ridge, w_bar, label="Ridge", color=C_PROJ,
           edgecolor="white", linewidth=0.5)
    ax.bar(x_pos + w_bar, w_lasso, w_bar, label="Lasso", color=C_WARN,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=C_FG, linewidth=0.6, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("coeficiente", fontsize=10, color=C_FG)
    ax.set_title("OLS vs Ridge vs Lasso (verdad dispersa)", fontsize=11,
                 color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_lasso_sparse_coefficients.png")


# ------------------------------------------------------------------
# 7a) Polynomial under/over-fitting
# ------------------------------------------------------------------
def fig_poly_underfit_overfit() -> None:
    rng = np.random.default_rng(0)
    n = 30
    x = np.linspace(-3, 3, n)
    y_true = 0.5 * x ** 3 - 0.8 * x
    y = y_true + rng.normal(0, 2.0, n)
    xgrid = np.linspace(-3.2, 3.2, 300)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.scatter(x, y, s=25, c=C_FG, alpha=0.7, zorder=5, label="datos")
    for deg, c, ls in [(1, C_RES, "--"), (3, C_PROJ, "-"), (12, C_WARN, "-.")]:
        V = np.vander(x, N=deg + 1, increasing=True)
        w, *_ = np.linalg.lstsq(V, y, rcond=None)
        Vg = np.vander(xgrid, N=deg + 1, increasing=True)
        ax.plot(xgrid, Vg @ w, color=c, linewidth=1.8, linestyle=ls,
                label=f"grado {deg}")
    ax.set_ylim(-18, 18)
    ax.set_xlabel("$x$", fontsize=10, color=C_FG)
    ax.set_ylabel("$y$", fontsize=10, color=C_FG)
    ax.set_title("Subajuste vs sobreajuste (polinomios)", fontsize=11,
                 color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_poly_underfit_overfit.png")


# ------------------------------------------------------------------
# 7b) Bias-variance curve (train/test error vs degree)
# ------------------------------------------------------------------
def fig_bias_variance_curve() -> None:
    rng = np.random.default_rng(0)
    n = 40
    x = np.linspace(-3, 3, n)
    y_true = 0.5 * x ** 3 - 0.8 * x
    y = y_true + rng.normal(0, 2.0, n)

    idx = rng.permutation(n)
    n_tr = 25
    xtr, ytr = x[idx[:n_tr]], y[idx[:n_tr]]
    xte, yte = x[idx[n_tr:]], y[idx[n_tr:]]

    degs = range(1, 16)
    train_err, test_err = [], []
    for deg in degs:
        Vtr = np.vander(xtr, N=deg + 1, increasing=True)
        w, *_ = np.linalg.lstsq(Vtr, ytr, rcond=None)
        train_err.append(float(np.mean((Vtr @ w - ytr) ** 2)))
        Vte = np.vander(xte, N=deg + 1, increasing=True)
        test_err.append(float(np.mean((Vte @ w - yte) ** 2)))

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(list(degs), train_err, "o-", color=C_ACCENT, linewidth=1.6,
            markersize=4, label="train")
    ax.plot(list(degs), test_err, "s-", color=C_RES, linewidth=1.6,
            markersize=4, label="test")
    ax.set_xlabel("grado del polinomio", fontsize=10, color=C_FG)
    ax.set_ylabel("MSE", fontsize=10, color=C_FG)
    ax.set_title("Error train/test vs complejidad", fontsize=11, color=C_FG,
                 pad=10)
    ax.set_yscale("log")
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)
    fig.tight_layout()
    _save_fig(fig, "week09_bias_variance_curve.png")


# ------------------------------------------------------------------
def main() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig_ols_line_and_residuals()
    fig_gd_loss_curve()
    fig_gd_lr_comparison()
    fig_ridge_vs_ols_coefs()
    fig_ridge_path()
    fig_lasso_sparse_coefs()
    fig_poly_underfit_overfit()
    fig_bias_variance_curve()


if __name__ == "__main__":
    main()
