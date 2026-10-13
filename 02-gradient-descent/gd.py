"""Gradient descent with nothing but NumPy.

gradient_descent: the whole algorithm. It works for one number or for many.
numerical_gradient: measures a slope by nudging the input. Used to check derivatives.
mse_gradient: the gradient of the mean squared error of a linear model.
minibatch_gradient_descent: the same loop, with each step computed on a sample of the rows."""

import numpy as np


def gradient_descent(grad, start, lr: float, steps: int) -> np.ndarray:
    """Returns every position visited, the start included."""
    x = np.array(start, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        # grad(x) is the slope at the current position.
        # Subtracting it moves x in the direction where the function gets smaller.
        x = x - lr * grad(x)
        path.append(x.copy())
        if not np.all(np.isfinite(x)):
            break
    return np.array(path)


def numerical_gradient(f, x, h: float = 1e-6) -> np.ndarray:
    """Change each input by a small amount h and measure how much f changes."""
    x = np.atleast_1d(np.array(x, dtype=float))
    slopes = np.zeros_like(x)
    for i in range(len(x)):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        slopes[i] = (f(up) - f(down)) / (2 * h)
    return slopes


def mse(X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
    w, b = params[:-1], params[-1]
    return float(np.mean((X @ w + b - y) ** 2))


def mse_gradient(X: np.ndarray, y: np.ndarray):
    """params holds the weights followed by the bias."""
    n = len(y)

    def grad(params):
        w, b = params[:-1], params[-1]
        # X @ w + b is the prediction for every row at once.
        # Subtracting y gives prediction minus actual: one error per row.
        error = X @ w + b - y
        # One slope per weight, then one for the bias.
        return np.append((2 / n) * (X.T @ error), 2 * error.mean())

    return grad


def minibatch_gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float, epochs: int, batch_size: int, seed: int = 0
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    params = np.zeros(d + 1)
    history = [mse(X, y, params)]
    for _ in range(epochs):
        # One epoch is one pass over all the rows, in a new random order.
        order = rng.permutation(n)
        for first in range(0, n, batch_size):
            rows = order[first : first + batch_size]
            params = params - lr * mse_gradient(X[rows], y[rows])(params)
        history.append(mse(X, y, params))
    return params, history
