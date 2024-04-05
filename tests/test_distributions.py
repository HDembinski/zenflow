import neural_flow.distributions as dist
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from jax.scipy.stats import multivariate_normal
import jax


def test_Uniform():
    uni = dist.Uniform()

    assert uni.bound == 5

    x = jnp.zeros((10, 3))
    lp = uni.log_prob(x)

    density = 1 / (2 * uni.bound) ** 3

    assert lp.shape == (10,)
    assert_allclose(lp, np.log(density))

    x = uni.sample(2, jax.random.PRNGKey(0))
    assert x.shape == (2, 3)
    assert np.min(x) >= -uni.bound
    assert np.max(x) <= uni.bound


def test_Normal():
    normal = dist.Normal()

    x = jnp.zeros((10, 3))
    lp = normal.log_prob(x)

    assert_allclose(normal._mean, np.zeros(3))
    assert_allclose(normal._cov, np.identity(3))

    assert_allclose(lp, multivariate_normal.logpdf(x, normal._mean, normal._cov))

    x = normal.sample(20000, jax.random.PRNGKey(0))
    assert x.shape == (20000, 3)

    assert_allclose(x.mean(0), np.zeros(3), atol=5e-2)
    assert_allclose(np.cov(x.T), np.identity(3), atol=5e-2)
