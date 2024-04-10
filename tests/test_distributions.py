import neural_flow.distributions as dist
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_compare
from operator import gt, lt
from jax.scipy.stats import multivariate_normal, beta
import jax


def test_Uniform():
    uni = dist.Uniform()

    x = jnp.zeros((10, 3))
    lp = uni.log_prob(x)

    assert lp.shape == (10,)
    assert_allclose(lp, 0)

    x = uni.sample(2, jax.random.PRNGKey(0))
    assert x.shape == (2, 3)
    assert np.min(x) >= 0
    assert np.max(x) < 1


def test_Normal():
    d = dist.Normal()

    rng = np.random.default_rng(1)
    x = rng.uniform(size=(10, 3))
    lp = d.log_prob(x)

    mean = 0.5 * np.ones(3)
    cov = np.identity(3) * 0.1**2
    assert_allclose(lp, multivariate_normal.logpdf(x, mean, cov), atol=1e-6)

    x = d.sample(20000, jax.random.PRNGKey(0))
    assert x.shape == (20000, 3)

    assert_allclose(x.mean(0), 0.5, atol=5e-2)
    assert_allclose(np.cov(x.T), 0.1**2 * np.identity(3), atol=5e-2)


def test_TruncatedNormal():
    d = dist.TruncatedNormal()

    rng = np.random.default_rng(1)
    x = rng.uniform(size=(10, 3))
    lp = d.log_prob(x)

    mean = 0.5 * np.ones(3)
    cov = np.identity(3) * 0.1**2
    assert_allclose(lp, multivariate_normal.logpdf(x, mean, cov), atol=5e-6)

    x = d.sample(20000, jax.random.PRNGKey(0))
    assert x.shape == (20000, 3)

    assert_allclose(x.mean(0), 0.5, atol=5e-2)
    assert_allclose(np.cov(x.T), 0.1**2 * np.identity(3), atol=5e-2)


def test_Beta():
    d = dist.Beta()

    rng = np.random.default_rng(1)
    x = rng.uniform(size=(10, 3))
    lp = d.log_prob(x)

    assert_allclose(
        lp,
        beta.logpdf(x, 12, 12).sum(-1),
    )

    x = d.sample(20000, jax.random.PRNGKey(0))
    assert x.shape == (20000, 3)

    assert_allclose(x.mean(0), 0.5, atol=5e-2)
    assert_array_compare(gt, x, 0)
    assert_array_compare(lt, x, 1)
