import numpy as np
from sklearn.linear_model import LinearRegression

from linreg import closed_form, gradient_descent, normal_equation, r2, rmse


def make_data(seed=0, n=500, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.array([2.0, -1.0, 0.5, 3.0]) + 4.0 + rng.normal(scale=0.1, size=n)
    return X, y


def test_closed_form_matches_sklearn():
    X, y = make_data()
    w, b = closed_form(X, y)
    ref = LinearRegression().fit(X, y)
    assert np.allclose(w, ref.coef_) and np.isclose(b, ref.intercept_)


def test_normal_equation_matches_closed_form_on_well_behaved_data():
    X, y = make_data()
    w1, b1 = closed_form(X, y)
    w2, b2 = normal_equation(X, y)
    assert np.allclose(w1, w2) and np.isclose(b1, b2)


def test_gradient_descent_converges_to_the_same_weights():
    X, y = make_data()
    w, b, history = gradient_descent(X, y, lr=0.1, steps=2000)
    ref_w, ref_b = closed_form(X, y)
    assert np.allclose(w, ref_w, atol=1e-6) and np.isclose(b, ref_b, atol=1e-6)
    assert history[-1] < history[0]


def test_gradient_descent_diverges_with_a_learning_rate_that_is_too_high():
    X, y = make_data()
    _, _, history = gradient_descent(X, y, lr=1.5, steps=200)
    assert not np.isfinite(history[-1]) or history[-1] > history[0]


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0 and r2(y, y) == 1
    assert np.isclose(r2(y, np.full(3, y.mean())), 0)
