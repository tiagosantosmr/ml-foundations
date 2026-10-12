"""Runs every experiment in the linear regression post and writes
results.json plus the figures. Nothing in the post is typed by hand:
if a number is in the post, it came out of this script.

usage: python experiments.py [figures_dir]
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from linreg import add_bias, closed_form, gradient_descent, normal_equation, predict, r2, rmse

warnings.filterwarnings("ignore", category=RuntimeWarning)

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


def poly(x, degree, mu, sd):
    z = (x - mu) / sd
    return np.column_stack([z**k for k in range(1, degree + 1)])


data = fetch_california_housing()
X, y, names = data.data, data.target, list(data.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
mu, sd = X_tr.mean(0), X_tr.std(0)
Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
R = {"rows": len(X), "train_rows": len(X_tr), "test_rows": len(X_te), "features": names}

# --- one feature -----------------------------------------------------------
w1, b1 = closed_form(X_tr[:, [0]], y_tr)
p1 = predict(X_te[:, [0]], w1, b1)
R["one_feature"] = {
    "slope": float(w1[0]), "intercept": b1,
    "test_rmse": rmse(y_te, p1), "test_r2": r2(y_te, p1),
    "baseline_rmse": rmse(y_te, np.full_like(y_te, y_tr.mean())),
}
rng = np.random.default_rng(0)
show = rng.choice(len(X_tr), 3000, replace=False)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.scatter(X_tr[show, 0], y_tr[show], s=5, color=TEAL, alpha=0.35, linewidths=0)
xs = np.array([0, 15])
ax.plot(xs, w1[0] * xs + b1, color=AMBER, lw=2, label=f"value = {w1[0]:.3f} x income + {b1:.3f}")
ax.axhline(5.0, color=PLUM, lw=1, ls="--", label="values are capped at 5.0 ($500k)")
ax.set_xlabel("median income of the district (x $10k)")
ax.set_ylabel("median house value (x $100k)")
ax.set_xlim(0, 15); ax.set_ylim(0, 8)
ax.legend(loc="upper left")
save(fig, "one-feature-fit.png")

# --- all features: three solvers ------------------------------------------
w_cf, b_cf = closed_form(Z_tr, y_tr)
w_ne, b_ne = normal_equation(Z_tr, y_tr)
ref = LinearRegression().fit(Z_tr, y_tr)
p_cf = predict(Z_te, w_cf, b_cf)
t = time.perf_counter(); closed_form(Z_tr, y_tr); t_cf = time.perf_counter() - t
t = time.perf_counter(); gradient_descent(Z_tr, y_tr, lr=0.1, steps=1000); t_gd = time.perf_counter() - t
R["all_features"] = {
    "test_rmse": rmse(y_te, p_cf), "test_r2": r2(y_te, p_cf),
    "train_rmse": rmse(y_tr, predict(Z_tr, w_cf, b_cf)),
    "max_diff_vs_sklearn": float(max(np.abs(w_cf - ref.coef_).max(), abs(b_cf - ref.intercept_))),
    "max_diff_normal_equation": float(np.abs(w_ne - w_cf).max()),
    "weights": dict(zip(names, map(float, w_cf))), "bias": b_cf,
    "closed_form_seconds": t_cf, "gd_1000_steps_seconds": t_gd,
}

# --- the weighted sum, with made-up numbers ----------------------------------
# Illustration only: these values are invented so the arithmetic is easy to follow.
# Nothing here comes from the dataset, and nothing is written to results.json.
START = 40_000
ROWS = [("size", "1,500 sq ft", 1500, 150, "$150 per sq ft"),
        ("bedrooms", "3", 3, 10_000, "$10,000 each"),
        ("age", "20 years", 20, -1_000, "-$1,000 per year"),
        ("distance to centre", "10 miles", 10, -3_000, "-$3,000 per mile")]
total = START + sum(v * w for _, _, v, w, _ in ROWS)
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-1.9, len(ROWS) + 1.7)
COLS = [(0, "feature"), (27, "this home"), (47, "weight"), (75, "adds or removes")]
top = len(ROWS) + 1
for x, label in COLS:
    ax.text(x, top, label, color=FAINT, fontsize=8.5, va="center")
ax.plot([0, 100], [top - 0.5, top - 0.5], color=HAIRLINE, lw=1)
ax.text(0, top - 1, "starting amount", color=MUTED, fontsize=10, va="center")
ax.text(75, top - 1, f"${START:,}", color=INK, fontsize=10.5, va="center")
for i, (name, shown, v, w, wlabel) in enumerate(ROWS):
    yy = top - 2 - i
    amount = v * w
    colour = TEAL if amount >= 0 else AMBER
    ax.text(0, yy, name, color=INK, fontsize=10.5, va="center")
    ax.text(27, yy, shown, color=INK, fontsize=10.5, va="center")
    ax.text(42, yy, "x", color=FAINT, fontsize=10, va="center")
    ax.text(47, yy, wlabel, color=colour, fontsize=10.5, va="center")
    ax.text(70, yy, "=", color=FAINT, fontsize=10, va="center")
    ax.text(75, yy, f"{'+' if amount >= 0 else '-'}${abs(amount):,}", color=colour, fontsize=10.5,
            va="center", fontweight="bold")
ax.plot([0, 100], [-0.5, -0.5], color=FAINT, lw=1)
ax.text(0, -1.2, "predicted price", color=INK, fontsize=11, va="center", fontweight="bold")
ax.text(75, -1.2, f"${total:,}", color=INK, fontsize=12, va="center", fontweight="bold")
ax.set_title("made-up numbers, to show the calculation", fontsize=9, color=MUTED, loc="left", pad=8)
save(fig, "weighted-sum-example.png")

# --- gradient descent: learning rate --------------------------------------
best_mse = float(np.mean((predict(Z_tr, w_cf, b_cf) - y_tr) ** 2))
eig = np.linalg.eigvalsh(Z_tr.T @ Z_tr / len(Z_tr))
R["gd"] = {"best_mse": best_mse, "lr_limit": float(1 / eig.max()), "runs": {}}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for lr, color in [(0.01, PLUM), (0.1, TEAL), (0.3, AMBER), (0.5, "#e06c75")]:
    w, b, hist = gradient_descent(Z_tr, y_tr, lr=lr, steps=5000)
    reached = next((i for i, v in enumerate(hist) if v - best_mse < 1e-6), None)
    R["gd"]["runs"][str(lr)] = {
        "final_mse": hist[-1] if np.isfinite(hist[-1]) else None,
        "steps_to_converge": reached,
        "max_weight_diff": float(np.abs(w - w_cf).max()) if np.isfinite(hist[-1]) and hist[-1] < 10 else None,
    }
    R["gd"]["runs"][str(lr)].update(
        {"lowest_mse": float(np.min(hist)), "lowest_mse_step": int(np.argmin(hist)), "last_mse": float(hist[-1])}
    )
    h = np.array(hist[:600])
    ax.plot(np.arange(len(h)), h, color=color, lw=1.8, label=f"learning rate {lr}")
ax.axhline(best_mse, color=FAINT, lw=1, ls="--", label=f"closed-form minimum ({best_mse:.4f})")
ax.set_yscale("log"); ax.set_ylim(0.4, 1e4)
ax.set_xlabel("step"); ax.set_ylabel("training MSE (log scale)")
ax.legend(loc="upper right")
save(fig, "gd-learning-rates.png")

# --- gradient descent: unscaled features ----------------------------------
raw = {}
for lr in [0.1, 1e-3, 1e-6, 1e-7]:
    _, _, hist = gradient_descent(X_tr, y_tr, lr=lr, steps=5000)
    raw[str(lr)] = {"steps_before_overflow": len(hist) if not np.isfinite(hist[-1]) else None,
                    "final_mse": hist[-1] if np.isfinite(hist[-1]) else None}
Xb = add_bias(X_tr)
R["unscaled"] = {
    "runs": raw, "target_variance": float(y_tr.var()),
    "condition_number_raw": float(np.linalg.cond(Xb.T @ Xb)),
    "condition_number_standardized": float(eig.max() / eig.min()),
    "feature_std": dict(zip(names, map(float, X_tr.std(0)))),
}

# --- residuals and the price cap ------------------------------------------
res = y_te - p_cf
capped = y_te >= 5.0
R["residuals"] = {
    "capped_districts": int(capped.sum()), "mean_residual_capped": float(res[capped].mean()),
    "mean_residual_rest": float(res[~capped].mean()),
    "negative_predictions": int((p_cf < 0).sum()), "max_prediction": float(p_cf.max()),
}
edges = np.quantile(p_cf, [0, 0.25, 0.5, 0.75, 1])
R["residuals"]["std_by_prediction_quarter"] = [
    float(res[(p_cf >= lo) & (p_cf <= hi)].std()) for lo, hi in zip(edges[:-1], edges[1:])
]
w_raw, b_raw = closed_form(X_tr, y_tr)
R["checks"] = {
    "raw_vs_scaled_max_prediction_diff": float(np.abs(predict(X_te, w_raw, b_raw) - p_cf).max()),
    "mean_training_target": float(y_tr.mean()),
    "corr_rooms_bedrooms": float(np.corrcoef(X_tr[:, 2], X_tr[:, 3])[0, 1]),
}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.scatter(p_cf[~capped], y_te[~capped], s=5, color=TEAL, alpha=0.35, linewidths=0, label="test districts")
ax.scatter(p_cf[capped], y_te[capped], s=7, color=AMBER, alpha=0.8, linewidths=0, label="capped at $500k")
ax.plot([0, 8], [0, 8], color=FAINT, lw=1, ls="--", label="perfect prediction")
ax.set_xlim(-0.5, 8); ax.set_ylim(0, 5.5)
ax.set_xlabel("predicted value (x $100k)"); ax.set_ylabel("actual value (x $100k)")
ax.legend(loc="lower right")
save(fig, "predicted-vs-actual.png")

# --- overfitting: polynomial degree on 30 points --------------------------
x_all, x_val = X_tr[:, 0], X_te[:, 0]
degrees, n_points, n_samples = list(range(1, 13)), 30, 200
tr = np.zeros((n_samples, len(degrees))); va = np.zeros_like(tr)
for s in range(n_samples):
    idx = np.random.default_rng(s).choice(len(x_all), n_points, replace=False)
    xs_, ys_ = x_all[idx], y_tr[idx]
    m_, s_ = xs_.mean(), xs_.std()
    inside = (x_val >= xs_.min()) & (x_val <= xs_.max())
    for j, deg in enumerate(degrees):
        w, b = closed_form(poly(xs_, deg, m_, s_), ys_)
        tr[s, j] = rmse(ys_, predict(poly(xs_, deg, m_, s_), w, b))
        va[s, j] = rmse(y_te[inside], predict(poly(x_val[inside], deg, m_, s_), w, b))
R["overfitting"] = {
    "points": n_points, "samples": n_samples, "degrees": degrees,
    "train_rmse_median": list(map(float, np.median(tr, 0))),
    "val_rmse_median": list(map(float, np.median(va, 0))),
    "best_degree_counts": list(map(int, np.bincount(va.argmin(1) + 1, minlength=13)[1:])),
}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(degrees, np.median(tr, 0), color=TEAL, lw=2, marker="o", ms=4, label="training RMSE")
ax.plot(degrees, np.median(va, 0), color=AMBER, lw=2, marker="o", ms=4, label="validation RMSE")
ax.set_yscale("log"); ax.set_xticks(degrees)
ax.set_xlabel("polynomial degree"); ax.set_ylabel("RMSE, median of 200 samples (log scale)")
ax.legend(loc="upper left")
save(fig, "overfitting-curve.png")

# one sample, three fits
idx = np.random.default_rng(0).choice(len(x_all), n_points, replace=False)
xs_, ys_ = x_all[idx], y_tr[idx]
m_, s_ = xs_.mean(), xs_.std()
grid = np.linspace(xs_.min(), xs_.max(), 400)
one = {}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.scatter(xs_, ys_, s=22, color=INK, zorder=3, label="30 training districts")
for deg, color in [(1, TEAL), (3, PLUM), (9, AMBER)]:
    w, b = closed_form(poly(xs_, deg, m_, s_), ys_)
    ax.plot(grid, predict(poly(grid, deg, m_, s_), w, b), color=color, lw=1.8, label=f"degree {deg}")
    one[str(deg)] = {
        "train_rmse": rmse(ys_, predict(poly(xs_, deg, m_, s_), w, b)),
        "prediction_at_income_15": float(predict(poly(np.array([15.0]), deg, m_, s_), w, b)[0]),
    }
ax.set_ylim(0, 6)
ax.set_xlabel("median income (x $10k)"); ax.set_ylabel("median house value (x $100k)")
ax.legend(loc="upper left")
save(fig, "polynomial-fits.png")
R["overfitting"]["one_sample"] = {"income_range": [float(xs_.min()), float(xs_.max())], "fits": one}

# more data changes the answer
m_, s_ = x_all.mean(), x_all.std()
full = {}
for deg in [1, 3, 10]:
    w, b = closed_form(poly(x_all, deg, m_, s_), y_tr)
    full[str(deg)] = rmse(y_te, predict(poly(x_val, deg, m_, s_), w, b))
R["overfitting"]["all_rows_test_rmse"] = full

# where the textbook formula breaks
idx = np.random.default_rng(0).choice(len(x_all), n_points, replace=False)
xs_, ys_ = x_all[idx], y_tr[idx]
m_, s_ = xs_.mean(), xs_.std()
cond = {}
for deg in [1, 5, 10, 15]:
    P = poly(xs_, deg, m_, s_)
    w_a, b_a = closed_form(P, ys_)
    w_b, b_b = normal_equation(P, ys_)
    Pb = add_bias(P)
    cond[str(deg)] = {
        "condition_number": float(np.linalg.cond(Pb.T @ Pb)),
        "train_rmse_lstsq": rmse(ys_, predict(P, w_a, b_a)),
        "train_rmse_normal_equation": rmse(ys_, predict(P, w_b, b_b)),
    }
R["conditioning"] = cond

(HERE / "results.json").write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=2))
