import jax.numpy as jnp
from numpy.testing import assert_allclose
from neural_flow import utils


def test_rational_quadratic_spline():
    x = jnp.linspace(-1, 1, 10).reshape(1, -1)
    W = jnp.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 1, -1)
    H = jnp.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 1, -1)
    D = jnp.array([1.0, 1.0, 1.0]).reshape(1, 1, -1)
    B = 1.0
    y, log_det = utils.rational_quadratic_spline(x, W, H, D, B, False)
    assert_allclose(y, x)


def test_index():
    x = jnp.array([-2, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5]).reshape(1, -1)
    xk = jnp.array([-1, 0, 1]).reshape(1, 3)
    expected = []
    for xi in x[0]:
        if xi < xk[0][0]:
            expected.append(0)
        elif xk[0][-1] <= xi:
            expected.append(2)
        else:
            for j in range(len(xk[0]) - 1):
                if xk[0][j] <= xi < xk[0][j + 1]:
                    expected.append(j)
                    break
    ind, oob = utils._index(x, xk, 1)
    assert_allclose(ind[0, :, 0], expected)


def test_knots():
    dx = jnp.array((0.5, 0.5, 0.5))
    bound = jnp.sum(dx) / 2
    xk = utils._knots(dx, bound)
    assert_allclose(xk, [-0.75, -0.25, 0.25, 0.75])
