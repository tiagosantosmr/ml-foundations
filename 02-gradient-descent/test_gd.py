import numpy as np
from sklearn.linear_model import LinearRegression

from gd import gradient_descent, minibatch_gradient_descent, mse, mse_gradient, numerical_gradient


def make_data(seed=0, n=500, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.array([2.0, -1.0, 0.5, 3.0]) + 4.0 + rng.normal(scale=0.1, size=n)
    return X, y


def test_one_number_reaches_the_minimum():
    path = gradient_descent(lambda x: 2 * (x - 3), 0.0, lr=0.1, steps=200)
    assert np.isclose(path[-1], 3.0)


def test_a_learning_rate_that_is_too_high_moves_away_from_the_minimum():
    path = gradient_descent(lambda x: 2 * (x - 3), 0.0, lr=1.1, steps=50)
    assert abs(path[-1] - 3) > abs(path[0] - 3)


def test_the_start_decides_which_minimum_is_reached():
    slope = lambda x: 4 * x**3 - 6 * x + 1
    left = gradient_descent(slope, -2.0, lr=0.02, steps=500)[-1]
    right = gradient_descent(slope, 2.0, lr=0.02, steps=500)[-1]
    assert left < 0 < right


def test_mse_gradient_matches_the_numerical_gradient():
    X, y = make_data()
    params = np.random.default_rng(1).normal(size=5)
    measured = numerical_gradient(lambda p: mse(X, y, p), params)
    assert np.allclose(mse_gradient(X, y)(params), measured, atol=1e-5)


def test_linear_regression_reaches_the_sklearn_weights():
    X, y = make_data()
    params = gradient_descent(mse_gradient(X, y), np.zeros(5), lr=0.1, steps=2000)[-1]
    ref = LinearRegression().fit(X, y)
    assert np.allclose(params[:-1], ref.coef_, atol=1e-6) and np.isclose(params[-1], ref.intercept_, atol=1e-6)


def test_minibatches_get_close_to_the_same_weights():
    X, y = make_data()
    params, history = minibatch_gradient_descent(X, y, lr=0.01, epochs=50, batch_size=32)
    ref = LinearRegression().fit(X, y)
    assert np.allclose(params[:-1], ref.coef_, atol=0.05) and history[-1] < history[0]
