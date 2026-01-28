import math
import os

import matplotlib
import numpy as np

# Non-interactive backend for CI / headless runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTDIR = os.path.join("slides", "figs")
os.makedirs(OUTDIR, exist_ok=True)


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    # PDF(x) = x^(a-1) (1-x)^(b-1) / B(a,b)
    lb = log_beta(a, b)
    return np.exp((a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - lb)


def main() -> None:
    x = np.linspace(1e-4, 1.0 - 1e-4, 1500)

    priors = [
        ((1, 1), r"Beta(1,1)  (uniforme)"),
        ((2, 2), r"Beta(2,2)  (suave, centrado en 0.5)"),
        ((8, 2), r"Beta(8,2)  (sesgo a $\theta$ altos)"),
    ]

    plt.figure(figsize=(7.4, 4.2))
    for (a, b), label in priors:
        y = beta_pdf(x, a, b)
        plt.plot(x, y, linewidth=2.2, label=label)

    plt.title(r"Priors Beta sobre $\theta$ (probabilidad de éxito)")
    plt.xlabel(r"$\theta$")
    plt.ylabel("densidad")
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    out = os.path.join(OUTDIR, "week02_beta_priors.png")
    plt.savefig(out, dpi=220)
    plt.close()
    print("saved:", out)


if __name__ == "__main__":
    main()

