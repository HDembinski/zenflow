"""Define the latent distributions used in the normalizing flows."""

from abc import ABC, abstractmethod
from .typing import Array
from typing import Any
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal


class Distribution(ABC):
    """Base class for latent distributions."""

    x_dim: int = 0

    def init(self, x: Array):
        self.x_dim = x.shape[1]
        assert self.x_dim > 0

    @abstractmethod
    def log_prob(self, X: Array) -> Array:
        """Calculate log-probability of the inputs."""

    @abstractmethod
    def sample(self, nsamples: int, seed: int = None) -> Array:
        """Sample from the distribution."""


class Normal(Distribution):
    """
    A multivariate Gaussian distribution with mean zero and unit variance.

    Note this distribution has infinite support, so it is not recommended that
    you use it with the spline coupling layers, which have compact support.
    If you do use the two together, you should set the support of the spline
    layers (using the spline parameter B) to be large enough that you rarely
    draw Gaussian samples outside the support of the splines.
    """

    def init(self, x: Array) -> None:
        super().init(x)
        self._mean = jnp.zeros(self.x_dim)
        self._cov = jnp.identity(self.x_dim)

    def log_prob(self, X: Array) -> Array:
        return multivariate_normal.logpdf(
            X,
            mean=self._mean,
            cov=self._cov,
        )

    def sample(self, nsamples: int, rngkey: Any = None) -> Array:
        return random.multivariate_normal(
            key=rngkey,
            mean=self._mean,
            cov=self._cov,
            shape=(nsamples,),
        )


class Uniform(Distribution):
    """A multivariate uniform distribution with support [-bound, bound]."""

    bound: float

    def __init__(self, bound: float = 5) -> None:
        self.bound = bound

    def log_prob(self, X: Array) -> Array:
        # which inputs are inside the support of the distribution
        mask = jnp.prod((X >= -self.bound) & (X <= self.bound), axis=-1)

        # calculate log_prob
        log_prob = jnp.where(
            mask,
            -self.x_dim * jnp.log(2 * self.bound),
            -jnp.inf,
        )

        return log_prob

    def sample(self, nsamples: int, rngkey: Any = None) -> Array:
        samples = random.uniform(
            rngkey,
            shape=(nsamples, self.x_dim),
            minval=-self.bound,
            maxval=self.bound,
        )
        return samples
