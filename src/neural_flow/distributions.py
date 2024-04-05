"""Base distributions used in conditional normalizing flows."""

from abc import ABC, abstractmethod
from jaxtyping import Array
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal


class Distribution(ABC):
    """Distribution base class with infrastructure for lazy initialization."""

    _initialized: bool = False

    def log_prob(self, x: Array) -> Array:
        """
        Compute log-probability of the samples.

        Parameters
        ----------
        x : Array of shape (N, D)
            N samples from the D-dimensional distribution.

        Returns
        -------
        log_prob : Array of shape (N,)
            Log-probabilities of the samples.

        """
        if not self._initialized:
            self._post_init(x.shape[-1])
            self._initialized = True
        return self._log_prob_impl(x)

    @abstractmethod
    def _post_init(self, dim: int) -> None: ...

    @abstractmethod
    def _log_prob_impl(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, nsamples: int, rngkey: Array) -> Array: ...


class Normal(Distribution):
    """
    A multivariate Gaussian distribution with mean zero and unit variance.

    Note this distribution has infinite support, so it is not recommended that
    you use it with the spline coupling layers, which have compact support.
    """

    _mean: Array
    _cov: Array

    def _post_init(self, dim: int) -> None:
        self._mean = jnp.zeros(dim)
        self._cov = jnp.identity(dim)

    def _log_prob_impl(self, x: Array) -> Array:
        return multivariate_normal.logpdf(
            x,
            mean=self._mean,
            cov=self._cov,
        )

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.multivariate_normal(
            key=rngkey,
            mean=self._mean,
            cov=self._cov,
            shape=(nsamples,),
        )


class Uniform(Distribution):
    """A multivariate uniform distribution with support [-bound, bound]."""

    bound: float
    _log_prob_const: float

    def __init__(self, bound: float = 5):
        self.bound = bound

    def _post_init(self, dim: int) -> None:
        self._dim = dim
        self._log_prob_const = -dim * jnp.log(2 * self.bound)

    def _log_prob_impl(self, x: Array) -> Array:
        # which inputs are inside the support of the distribution
        mask = jnp.prod((x >= -self.bound) & (x <= self.bound), axis=-1)
        return jnp.where(
            mask,
            self._log_prob_const,
            -jnp.inf,
        )

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.uniform(
            rngkey,
            shape=(nsamples, self._dim),
            minval=-self.bound,
            maxval=self.bound,
        )
