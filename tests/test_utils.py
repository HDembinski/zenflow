import jax.numpy as jnp
from numpy.testing import assert_allclose
from neural_flow import utils


def test_rational_quadratic_spline():
    x = jnp.linspace(-1, 1, 10).reshape(1, -1)
    W = jnp.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 1, -1)
    H = jnp.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 1, -1)
    D = jnp.array([1.0, 1.0, 1.0]).reshape(1, 1, -1)
    B = 1.0
    y, log_det = utils.rational_quadratic_spline(x, W, H, D, B, False, False)
    assert_allclose(y, x)


# def test_index():
#     x = jnp.linspace(-2, 2, 6).reshape(1, 6)
#     xk = jnp.array([-1, 0, 1]).reshape(1, 3)
#     i = _index(x, xk)
#     print(i)
#     assert i.shape == (1, 6)
#     assert_allclose(i, [[0, 0, 1, 2, 2, 2]])
