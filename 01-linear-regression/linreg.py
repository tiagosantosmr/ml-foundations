"""Linear regression two ways, with nothing but NumPy.

closed_form: solves for the weights directly.
gradient_descent: starts at zero and walks downhill on the squared error.
Both return (weights, bias)."""

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def closed_form(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    # lstsq instead of inv(X.T @ X): same answer when X is well behaved,
    # and it does not fall apart when columns are nearly redundant.
    theta, *_ = np.linalg.lstsq(add_bias(X), y, rcond=None)
    return theta[1:], float(theta[0])


def normal_equation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """The textbook formula, inverse and all. Kept to show where it breaks."""
    Xb = add_bias(X)
    theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
    return theta[1:], float(theta[0])


def gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000
) -> tuple[np.ndarray, float, list[float]]:
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    history = []
    for _ in range(steps):
        error = X @ w + b - y
        history.append(float(np.mean(error**2)))
        if not np.isfinite(history[-1]):
            break
        w -= lr * (2 / n) * (X.T @ error)
        b -= lr * 2 * float(np.mean(error))
    return w, b, history


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
