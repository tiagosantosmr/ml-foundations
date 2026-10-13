"""Runs every experiment in the gradient descent post and writes
results.json plus the figures. Nothing in the post is typed by hand:
if a number is in the post, it came out of this script.

usage: python experiments.py [figures_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from gd import gradient_descent, minibatch_gradient_descent, mse, mse_gradient, numerical_gradient

np.seterr(all="ignore")

HERE = Path(__file__).parent
FIG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SURFACE, HAIRLINE, INK, MUTED, FAINT = "#0f172a", "#1e293b", "#f1f5f9", "#94a3b8", "#475569"
TEAL, AMBER, PLUM = "#40aec9", "#e3a542", "#9b87c4"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": HAIRLINE, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": HAIRLINE,
    "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans Mono", "font.size": 9, "legend.frameon": False,
    "legend.labelcolor": MUTED, "figure.dpi": 200,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)


def steps_to(path, target, tol=1e-3):
    """First step at which every value is within tol of the target."""
    for i, p in enumerate(path):
        if np.all(np.isfinite(p)) and np.max(np.abs(p - target)) < tol:
            return i
    return None


R = {}

# --- one number: f(x) = (x - 3)^2 ------------------------------------------
f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)

path = gradient_descent(df, 0.0, lr=0.1, steps=50)
R["one_number"] = {
    "start": 0.0, "minimum": 3.0, "lr": 0.1,
    "numerical_slope_at_start": float(numerical_gradient(lambda v: f(v[0]), 0.0)[0]),
    "first_steps": [
        {"x": float(x), "slope": float(df(x)), "move": float(-0.1 * df(x)), "new_x": float(x - 0.1 * df(x))}
        for x in path[:5]
    ],
    "x_after_10": float(path[10]), "x_after_50": float(path[50]),
}

xs = np.linspace(-0.6, 6.6, 300)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(xs, f(xs), color=FAINT, lw=1.5)
shown = path[:11]
ax.plot(shown, f(shown), color=TEAL, lw=1, alpha=0.6)
ax.scatter(shown, f(shown), color=TEAL, s=22, zorder=3)
ax.scatter([3], [0], color=AMBER, s=40, zorder=4, label="minimum, x = 3")
ax.annotate("start, x = 0", (0, 9), xytext=(0.5, 10.5), color=MUTED, arrowprops={"arrowstyle": "-", "color": FAINT})
ax.set_xlabel("x")
ax.set_ylabel("f(x) = (x - 3)²")
ax.set_title("The first 10 steps with a learning rate of 0.1", loc="left", fontsize=10)
ax.legend(loc="upper center")
save(fig, "first-steps.png")

# --- learning rates on the same curve --------------------------------------
R["learning_rates"] = {}
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True)
wide = np.linspace(-4.5, 10.5, 300)
for ax, lr in zip(axes.ravel(), (0.01, 0.1, 0.9, 1.1)):
    p = gradient_descent(df, 0.0, lr, steps=1000)
    R["learning_rates"][str(lr)] = {
        "x_after_1": float(p[1]), "x_after_2": float(p[2]), "x_after_10": float(p[10]),
        "x_after_20": float(p[20]), "steps_to_minimum": steps_to(p, 3.0),
        "distance_multiplier": 1 - 2 * lr,
    }
    ax.plot(wide, f(wide), color=FAINT, lw=1.5)
    shown = p[:11]
    color = PLUM if lr > 1 else TEAL
    ax.plot(shown, f(shown), color=color, lw=1, alpha=0.6)
    ax.scatter(shown, f(shown), color=color, s=16, zorder=3)
    ax.scatter([3], [0], color=AMBER, s=30, zorder=4)
    ax.set_title(f"learning rate {lr}", loc="left", fontsize=9, color=MUTED)
    ax.set_xlim(-4.5, 10.5)
    ax.set_ylim(-3, 56)
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("f(x)")
save(fig, "learning-rates.png")

# --- two valleys: g(x) = x^4 - 3x^2 + x ------------------------------------
g = lambda x: x**4 - 3 * x**2 + x
dg = lambda x: 4 * x**3 - 6 * x + 1
R["two_valleys"] = {"lr": 0.02, "runs": {}}
xs = np.linspace(-2.1, 2.1, 400)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(xs, g(xs), color=FAINT, lw=1.5)
for start, color in ((-2.0, TEAL), (2.0, PLUM)):
    p = gradient_descent(dg, start, lr=0.02, steps=500)
    R["two_valleys"]["runs"][str(start)] = {
        "end_x": float(p[-1]), "end_value": float(g(p[-1])), "steps": steps_to(p, p[-1]),
    }
    shown = p[:40]
    ax.plot(shown, g(shown), color=color, lw=1, alpha=0.6)
    ax.scatter(shown, g(shown), color=color, s=14, zorder=3,
               label=f"start at {start:g}: ends at x = {p[-1]:.2f}, g = {g(p[-1]):.2f}")
ax.set_xlabel("x")
ax.set_ylabel("g(x) = x⁴ - 3x² + x")
ax.legend(loc="upper center")
save(fig, "two-valleys.png")

# --- a model: house value from income, two parameters ----------------------
data = fetch_california_housing()
X, y, names = data.data, data.target, list(data.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
income = X_tr[:, [0]]
ref = LinearRegression().fit(income, y_tr)
direct = np.array([ref.coef_[0], ref.intercept_])

rng = np.random.default_rng(0)
probe = rng.normal(size=2)
grad_check = float(np.max(np.abs(
    mse_gradient(income, y_tr)(probe) - numerical_gradient(lambda p: mse(income, y_tr, p), probe)
)))

second_moment = np.array([[np.mean(income**2), income.mean()], [income.mean(), 1.0]])
raw_runs, raw_paths = {}, {}
for lr in (0.01, 0.05, 0.06):
    p = gradient_descent(mse_gradient(income, y_tr), [0.0, 0.0], lr, steps=5000)
    ok = bool(np.all(np.isfinite(p[-1])))
    raw_paths[lr] = p
    raw_runs[str(lr)] = {
        "steps_to_direct_solution": steps_to(p, direct),
        "slope": float(p[-1][0]) if ok else None, "intercept": float(p[-1][1]) if ok else None,
        "max_diff": float(np.max(np.abs(p[-1] - direct))) if ok else None,
        "steps_before_overflow": None if ok else len(p) - 1,
    }

mu1, sd1 = float(income.mean()), float(income.std())
z = (income - mu1) / sd1
ref_z = LinearRegression().fit(z, y_tr)
direct_z = np.array([ref_z.coef_[0], ref_z.intercept_])
std_runs, std_paths = {}, {}
for lr in (0.01, 0.05, 0.4):
    p = gradient_descent(mse_gradient(z, y_tr), [0.0, 0.0], lr, steps=5000)
    std_paths[lr] = p
    # Convert back to the raw feature to compare with the first model.
    slope_raw = p[-1][0] / sd1
    std_runs[str(lr)] = {
        "steps_to_direct_solution": steps_to(p, direct_z),
        "slope_standardized": float(p[-1][0]), "intercept_standardized": float(p[-1][1]),
        "slope_raw_units": float(slope_raw), "intercept_raw_units": float(p[-1][1] - slope_raw * mu1),
    }

R["model"] = {
    "train_rows": len(income),
    "direct": {"slope": float(direct[0]), "intercept": float(direct[1])},
    "gradient_check_max_diff": grad_check,
    "start_mse": mse(income, y_tr, np.zeros(2)), "best_mse": mse(income, y_tr, direct),
    "income_mean": mu1, "income_std": sd1,
    "raw": {"lr_limit": float(1 / np.linalg.eigvalsh(second_moment)[-1]), "runs": raw_runs},
    "standardized": {"runs": std_runs},
}


def bowl(ax, x, target, path, w_range, b_range, title):
    ws, bs = np.meshgrid(np.linspace(*w_range, 200), np.linspace(*b_range, 200))
    loss = (ws**2 * np.mean(x**2) + bs**2 + np.mean(y_tr**2) + 2 * ws * bs * x.mean()
            - 2 * ws * np.mean(x[:, 0] * y_tr) - 2 * bs * y_tr.mean())
    best = mse(x, y_tr, target)
    ax.contour(ws, bs, loss, levels=best + np.geomspace(0.02, 40, 12), colors=FAINT, linewidths=0.8)
    ax.plot(path[:, 0], path[:, 1], color=TEAL, lw=1)
    ax.scatter(path[:, 0], path[:, 1], color=TEAL, s=6, zorder=3)
    ax.scatter(*target, color=AMBER, s=40, zorder=4)
    ax.set_xlim(*w_range)
    ax.set_ylim(*b_range)
    ax.set_xlabel("weight")
    ax.set_ylabel("bias")
    ax.set_title(title, loc="left", fontsize=9, color=MUTED)
    ax.grid(False)


n_raw = raw_runs["0.01"]["steps_to_direct_solution"]
n_std = std_runs["0.01"]["steps_to_direct_solution"]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
bowl(axes[0], income, direct, raw_paths[0.01][: n_raw + 1], (-0.1, 0.9), (-0.5, 1.0),
     f"raw income: {n_raw:,} steps")
bowl(axes[1], z, direct_z, std_paths[0.01][: n_std + 1], (-0.4, 1.6), (-0.4, 3.2),
     f"standardized income: {n_std} steps")
save(fig, "loss-bowl.png")

# --- all 8 features ---------------------------------------------------------
mu, sd = X_tr.mean(0), X_tr.std(0)
Z = (X_tr - mu) / sd
ref8 = LinearRegression().fit(Z, y_tr)
direct8 = np.append(ref8.coef_, ref8.intercept_)
p8 = gradient_descent(mse_gradient(Z, y_tr), np.zeros(9), lr=0.1, steps=2000)
R["all_features"] = {
    "features": names, "lr": 0.1, "steps": 2000,
    "steps_to_direct_solution": steps_to(p8, direct8),
    "max_diff": float(np.max(np.abs(p8[-1] - direct8))),
    "mse": mse(Z, y_tr, p8[-1]), "best_mse": mse(Z, y_tr, direct8),
}

# --- how many rows per step -------------------------------------------------
EPOCHS = 10
n = len(Z)
R["batches"] = {"epochs": EPOCHS, "best_mse": R["all_features"]["best_mse"], "runs": {},
                "largest_standardized_value": float(np.abs(Z).max())}
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for (size, lr), color in zip(((n, 0.1), (32, 0.01), (1, 0.0001)), (PLUM, TEAL, AMBER)):
    _, history = minibatch_gradient_descent(Z, y_tr, lr, EPOCHS, size)
    label = "all rows" if size == n else f"{size} row" + ("s" if size > 1 else "")
    R["batches"]["runs"][str(size)] = {
        "label": label, "lr": lr, "steps_per_pass": int(np.ceil(n / size)),
        "mse_after_1": history[1], "mse_after_10": history[10],
        "lowest_mse": float(min(history)), "highest_mse_after_first_pass": float(max(history[1:])),
    }
    ax.plot(range(EPOCHS + 1), history, color=color, lw=1.6, marker="o", ms=3,
            label=f"{label} per step, learning rate {lr:g}")
ax.axhline(R["batches"]["best_mse"], color=FAINT, lw=1, ls="--", label="minimum")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4])
ax.set_yticks([], minor=True)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("passes over the training data")
ax.set_ylabel("training MSE")
ax.legend()
save(fig, "batch-sizes.png")

# Larger learning rates for the smaller batches, to record that they fail.
R["batches"]["diverging"] = {}
for size, lr in ((32, 0.1), (1, 0.001)):
    _, history = minibatch_gradient_descent(Z, y_tr, lr, 2, size)
    R["batches"]["diverging"][str(size)] = {"lr": lr, "mse_after_1": history[1]}

(HERE / "results.json").write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=2))
