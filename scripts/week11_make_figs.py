"""
Week 11 — Logistic Regression figures (pedagogical Matplotlib).

Generates PNGs in slides/figs/ with prefix week11_*.
Uses only matplotlib + numpy + sklearn (no seaborn).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)

# Paleta consistente con el resto del curso.
C_FG = "#1f2937"
C_ACCENT = "#2563eb"      # azul
C_RES = "#dc2626"         # rojo
C_PROJ = "#059669"        # verde
C_GRID = "#e5e7eb"
C_WARN = "#d97706"        # naranja
C_PURPLE = "#7c3aed"


def _save_fig(fig, fname: str, *, pad_inches: float = 0.22) -> None:
    out = os.path.join(OUTDIR, fname)
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight",
                pad_inches=pad_inches)
    plt.close(fig)
    print("saved:", out)


def _style_ax(ax) -> None:
    ax.grid(True, linestyle=":", alpha=0.5, color=C_GRID)
    ax.tick_params(colors=C_FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_FG)


# ------------------------------------------------------------------
# 1) Scatter 2D: dos clases separables con frontera lineal
# ------------------------------------------------------------------
def fig_class_scatter() -> None:
    rng = np.random.default_rng(3)
    n = 80
    X0 = rng.normal(loc=[-1.2, -0.5], scale=0.7, size=(n, 2))
    X1 = rng.normal(loc=[1.2, 0.8], scale=0.7, size=(n, 2))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(X0[:, 0], X0[:, 1], c=C_ACCENT, s=28, edgecolors="white",
               linewidths=0.6, label="clase 0")
    ax.scatter(X1[:, 0], X1[:, 1], c=C_RES, s=28, edgecolors="white",
               linewidths=0.6, label="clase 1")

    # Frontera ilustrativa lineal.
    xs = np.linspace(-3.5, 3.5, 100)
    ax.plot(xs, -0.8 * xs + 0.2, color=C_FG, linewidth=1.6,
            linestyle="--", label="frontera de decision")

    ax.set_xlabel("$x_1$", fontsize=10, color=C_FG)
    ax.set_ylabel("$x_2$", fontsize=10, color=C_FG)
    ax.set_title("Clasificacion binaria en 2D", fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_class_scatter.png")


# ------------------------------------------------------------------
# 2) Regresion lineal aplicada a etiquetas 0/1 (falla)
# ------------------------------------------------------------------
def fig_linreg_for_classification() -> None:
    rng = np.random.default_rng(0)
    n0, n1 = 20, 20
    x0 = rng.normal(-1.5, 0.6, n0)
    x1 = rng.normal(1.5, 0.6, n1)
    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    # Agregar algunos puntos extremos para empujar la recta fuera de [0,1].
    x_ext = np.array([4.5, 5.0, 5.2])
    x = np.concatenate([x, x_ext])
    y = np.concatenate([y, np.ones(3)])

    X = np.column_stack([np.ones_like(x), x])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    xs = np.linspace(-4, 6, 200)
    ys_line = w[0] + w[1] * xs

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.scatter(x[y == 0], y[y == 0], c=C_ACCENT, s=32,
               edgecolors="white", linewidths=0.6, label="clase 0")
    ax.scatter(x[y == 1], y[y == 1], c=C_RES, s=32,
               edgecolors="white", linewidths=0.6, label="clase 1")
    ax.plot(xs, ys_line, color=C_PROJ, linewidth=2.0,
            label="recta de regresion lineal")
    ax.axhline(0, color=C_FG, linewidth=0.6, linestyle=":")
    ax.axhline(1, color=C_FG, linewidth=0.6, linestyle=":")
    ax.axhspan(-0.4, 0, alpha=0.08, color=C_WARN)
    ax.axhspan(1, 1.4, alpha=0.08, color=C_WARN)

    ax.set_xlabel("feature $x$", fontsize=10, color=C_FG)
    ax.set_ylabel("etiqueta / prediccion", fontsize=10, color=C_FG)
    ax.set_title("Regresion lineal sobre etiquetas 0/1: sale de $[0,1]$",
                 fontsize=11, color=C_FG, pad=10)
    ax.set_ylim(-0.4, 1.4)
    ax.legend(fontsize=8, framealpha=0.95, loc="lower right")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_linreg_for_classification.png")


# ------------------------------------------------------------------
# 3) Curva sigmoid
# ------------------------------------------------------------------
def fig_sigmoid() -> None:
    z = np.linspace(-10, 10, 400)
    s = 1.0 / (1.0 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(z, s, color=C_ACCENT, linewidth=2.2)
    ax.axhline(0.5, color=C_FG, linewidth=0.8, linestyle=":")
    ax.axvline(0, color=C_FG, linewidth=0.8, linestyle=":")
    ax.scatter([0], [0.5], color=C_RES, s=45, zorder=5)
    ax.annotate(r"$(0,\,0.5)$", xy=(0, 0.5), xytext=(1.0, 0.30),
                fontsize=9, color=C_FG,
                arrowprops=dict(arrowstyle="->", color=C_FG, lw=0.8))
    ax.set_xlabel(r"$z$", fontsize=10, color=C_FG)
    ax.set_ylabel(r"$\sigma(z)$", fontsize=10, color=C_FG)
    ax.set_title(r"Funcion sigmoid $\sigma(z) = 1/(1 + e^{-z})$",
                 fontsize=11, color=C_FG, pad=10)
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_sigmoid.png")


# ------------------------------------------------------------------
# 4) Frontera de decision real (logistic regression ajustada)
# ------------------------------------------------------------------
def fig_decision_boundary() -> None:
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=1.5, random_state=4,
    )
    clf = LogisticRegression().fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cf = ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.55,
                     vmin=0, vmax=1)
    ax.contour(xx, yy, probs, levels=[0.5], colors=[C_FG], linewidths=1.6)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=C_ACCENT, s=26,
               edgecolors="white", linewidths=0.6, label="clase 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C_RES, s=26,
               edgecolors="white", linewidths=0.6, label="clase 1")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"$p(y=1\mid x)$", fontsize=9, color=C_FG)
    cbar.ax.tick_params(colors=C_FG, labelsize=8)

    ax.set_xlabel("$x_1$", fontsize=10, color=C_FG)
    ax.set_ylabel("$x_2$", fontsize=10, color=C_FG)
    ax.set_title("Frontera de decision lineal y mapa de probabilidad",
                 fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_decision_boundary.png")


# ------------------------------------------------------------------
# 5) Cross-entropy loss: -log(p_hat) cuando y=1 (y espejo para y=0)
# ------------------------------------------------------------------
def fig_cross_entropy_loss() -> None:
    p = np.linspace(1e-3, 1.0 - 1e-3, 400)
    loss_y1 = -np.log(p)
    loss_y0 = -np.log(1.0 - p)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(p, loss_y1, color=C_RES, linewidth=2.0,
            label=r"$y=1$: $-\log(\hat p)$")
    ax.plot(p, loss_y0, color=C_ACCENT, linewidth=2.0,
            label=r"$y=0$: $-\log(1-\hat p)$")
    ax.axvline(0.5, color=C_FG, linewidth=0.6, linestyle=":")

    ax.set_xlabel(r"probabilidad predicha $\hat p$", fontsize=10, color=C_FG)
    ax.set_ylabel("perdida", fontsize=10, color=C_FG)
    ax.set_title("Binary cross-entropy por observacion",
                 fontsize=11, color=C_FG, pad=10)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_cross_entropy.png")


# ------------------------------------------------------------------
# 6) Logistic regression con GD: curvas de perdida para varios eta
# ------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fig_gd_learning_rate() -> None:
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=3, n_redundant=1,
        class_sep=1.2, random_state=1,
    )
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    Xt = np.column_stack([np.ones(n), X])

    iters = 200
    etas = [0.02, 0.2, 2.5]
    labels = [r"$\eta=0.02$ (lento)", r"$\eta=0.2$ (adecuado)",
              r"$\eta=2.5$ (inestable)"]
    colors = [C_PROJ, C_ACCENT, C_RES]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for eta, c, lab in zip(etas, colors, labels):
        theta = np.zeros(d + 1)
        losses = []
        for _ in range(iters):
            p = _sigmoid(Xt @ theta)
            losses.append(_bce(y, p))
            grad = Xt.T @ (p - y) / n
            theta -= eta * grad
        losses = np.array(losses)
        ax.plot(np.clip(losses, 0, 3.0), color=c, linewidth=1.8, label=lab)

    ax.set_xlabel("iteracion", fontsize=10, color=C_FG)
    ax.set_ylabel("binary cross-entropy", fontsize=10, color=C_FG)
    ax.set_title("Gradient descent: efecto de la tasa de aprendizaje",
                 fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_gd_learning_rate.png")


# ------------------------------------------------------------------
# 7) Batch GD vs SGD: curvas de perdida
# ------------------------------------------------------------------
def fig_gd_vs_sgd() -> None:
    X, y = make_classification(
        n_samples=400, n_features=4, n_informative=3, n_redundant=1,
        class_sep=1.2, random_state=2,
    )
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    Xt = np.column_stack([np.ones(n), X])

    # Batch GD
    theta = np.zeros(d + 1)
    eta = 0.2
    batch_loss = []
    for _ in range(80):
        p = _sigmoid(Xt @ theta)
        batch_loss.append(_bce(y, p))
        grad = Xt.T @ (p - y) / n
        theta -= eta * grad

    # SGD (mini-batch = 1), medimos perdida cada paso
    rng = np.random.default_rng(0)
    theta = np.zeros(d + 1)
    eta_sgd = 0.05
    sgd_loss = []
    idx = rng.permutation(n)
    for t in range(80):
        i = idx[t % n]
        xi = Xt[i]
        pi = _sigmoid(xi @ theta)
        grad = (pi - y[i]) * xi
        theta -= eta_sgd * grad
        p = _sigmoid(Xt @ theta)
        sgd_loss.append(_bce(y, p))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(batch_loss, color=C_ACCENT, linewidth=2.0, label="Batch GD")
    ax.plot(sgd_loss, color=C_RES, linewidth=1.6, label="SGD (mini-batch=1)")
    ax.set_xlabel("iteracion", fontsize=10, color=C_FG)
    ax.set_ylabel("binary cross-entropy", fontsize=10, color=C_FG)
    ax.set_title("Batch GD vs SGD (perdida total por paso)",
                 fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_gd_vs_sgd.png")


# ------------------------------------------------------------------
# 8) L1 vs L2: coeficientes aprendidos con features irrelevantes
# ------------------------------------------------------------------
def fig_l1_vs_l2() -> None:
    X, y = make_classification(
        n_samples=500, n_features=12, n_informative=4, n_redundant=0,
        n_repeated=0, class_sep=1.0, random_state=10,
    )
    X = StandardScaler().fit_transform(X)

    clf_l2 = LogisticRegression(penalty="l2", C=0.5, solver="liblinear",
                                max_iter=2000).fit(X, y)
    clf_l1 = LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                max_iter=2000).fit(X, y)
    w_l2 = clf_l2.coef_.ravel()
    w_l1 = clf_l1.coef_.ravel()

    d = X.shape[1]
    x_pos = np.arange(d)
    width = 0.4

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x_pos - width / 2, w_l2, width, label="L2 ($C=0.5$)",
           color=C_ACCENT, edgecolor="white", linewidth=0.6)
    ax.bar(x_pos + width / 2, w_l1, width, label="L1 ($C=0.5$)",
           color=C_PROJ, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=C_FG, linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$w_{{{j+1}}}$" for j in range(d)], fontsize=8)
    ax.set_ylabel("coeficiente", fontsize=10, color=C_FG)
    ax.set_title("Regularizacion L1 vs L2 en logistic regression",
                 fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95)
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_l1_vs_l2.png")


# ------------------------------------------------------------------
# 9) Precision / Recall / F1 vs threshold
# ------------------------------------------------------------------
def fig_threshold_tuning() -> None:
    X, y = make_classification(
        n_samples=1000, n_features=6, n_informative=4, n_redundant=1,
        weights=[0.8, 0.2], class_sep=1.1, random_state=5,
    )
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    probs = clf.predict_proba(X)[:, 1]

    ts = np.linspace(0.05, 0.95, 37)
    precs, recs, f1s = [], [], []
    for t in ts:
        yhat = (probs >= t).astype(int)
        precs.append(precision_score(y, yhat, zero_division=0))
        recs.append(recall_score(y, yhat, zero_division=0))
        f1s.append(f1_score(y, yhat, zero_division=0))
    t_star = ts[int(np.argmax(f1s))]

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(ts, precs, color=C_ACCENT, linewidth=2.0, label="precision")
    ax.plot(ts, recs, color=C_RES, linewidth=2.0, label="recall")
    ax.plot(ts, f1s, color=C_PROJ, linewidth=2.0, label="F1")
    ax.axvline(0.5, color=C_FG, linewidth=0.8, linestyle=":")
    ax.axvline(t_star, color=C_WARN, linewidth=1.2, linestyle="--",
               label=rf"$t^\star={t_star:.2f}$ (max F1)")

    ax.set_xlabel("threshold $t$", fontsize=10, color=C_FG)
    ax.set_ylabel("valor de la metrica", fontsize=10, color=C_FG)
    ax.set_title("Precision / Recall / F1 vs threshold",
                 fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95, loc="lower center")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_threshold_tuning.png")


# ------------------------------------------------------------------
# 10) Softmax: scores -> probabilidades (barra)
# ------------------------------------------------------------------
def fig_softmax_example() -> None:
    z = np.array([2.0, 1.0, -1.0])
    ez = np.exp(z - z.max())
    p = ez / ez.sum()
    labels = ["clase 1", "clase 2", "clase 3"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    axes[0].bar(labels, z, color=C_ACCENT, edgecolor="white", linewidth=0.6)
    axes[0].axhline(0, color=C_FG, linewidth=0.7)
    axes[0].set_title("scores $z_k$", fontsize=10, color=C_FG, pad=8)
    axes[0].set_ylabel("valor", fontsize=9, color=C_FG)
    _style_ax(axes[0])

    axes[1].bar(labels, p, color=C_PROJ, edgecolor="white", linewidth=0.6)
    for i, pi in enumerate(p):
        axes[1].text(i, pi + 0.02, f"{pi:.2f}", ha="center",
                     fontsize=9, color=C_FG)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(r"softmax: $p_k$ suman 1", fontsize=10, color=C_FG, pad=8)
    axes[1].set_ylabel("probabilidad", fontsize=9, color=C_FG)
    _style_ax(axes[1])

    fig.suptitle("Softmax convierte scores en probabilidades",
                 fontsize=11, color=C_FG, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "week11_softmax_example.png")


# ------------------------------------------------------------------
# 11) Multinomial logistic regression: 3 clases y regiones de decision
# ------------------------------------------------------------------
def fig_multiclass_regions() -> None:
    X, y = make_classification(
        n_samples=300, n_features=2, n_informative=2, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.4, random_state=7,
    )
    clf = LogisticRegression(max_iter=1000).fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(["#bfdbfe", "#fecaca", "#bbf7d0"])
    colors_pt = [C_ACCENT, C_RES, C_PROJ]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.contourf(xx, yy, Z, alpha=0.55, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])
    for k in range(3):
        ax.scatter(X[y == k, 0], X[y == k, 1], c=colors_pt[k], s=26,
                   edgecolors="white", linewidths=0.6, label=f"clase {k}")
    ax.set_xlabel("$x_1$", fontsize=10, color=C_FG)
    ax.set_ylabel("$x_2$", fontsize=10, color=C_FG)
    ax.set_title("Softmax / multinomial logistic regression (3 clases)",
                 fontsize=11, color=C_FG, pad=10)
    ax.legend(fontsize=8, framealpha=0.95, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    _save_fig(fig, "week11_multiclass_regions.png")


def main() -> None:
    fig_class_scatter()
    fig_linreg_for_classification()
    fig_sigmoid()
    fig_decision_boundary()
    fig_cross_entropy_loss()
    fig_gd_learning_rate()
    fig_gd_vs_sgd()
    fig_l1_vs_l2()
    fig_threshold_tuning()
    fig_softmax_example()
    fig_multiclass_regions()


if __name__ == "__main__":
    main()
