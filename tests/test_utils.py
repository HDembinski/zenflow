import jax.numpy as jnp
from numpy.testing import assert_allclose
from neural_flow.utils import rational_quadratic_spline


def test_rational_quadratic_spline():
    x = jnp.linspace(-1, 1, 10)
    W = jnp.array([0.5, 0.5, 0.5, 0.5])
    H = jnp.array([0.5, 0.5, 0.5, 0.5])
    D = jnp.array([0.0, 0.0, 0.0])
    B = 1.0
    y, log_det = rational_quadratic_spline(x, W, H, D, B, False, False)
    assert_allclose(y, 0)
