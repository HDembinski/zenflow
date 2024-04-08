"""Base distributions used in conditional normalizing flows."""

from abc import ABC, abstractmethod
from jaxtyping import Array
import jax.numpy as jnp
from jax import random
from jax.scipy import stats


class Distribution(ABC):
    """Distribution base class with infrastructure for lazy initialization."""

    __dim: int
    __initialized: bool = False

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
        if not self.__initialized:
            self.__dim = x.shape[-1]
            self.__initialized = True
        return self._log_prob_impl(x)

    @property
    def dim(self):
        return self.__dim

    @abstractmethod
    def _log_prob_impl(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, nsamples: int, rngkey: Array) -> Array: ...


class BoundedDistribution(Distribution):
    """Base class for bounded distributions."""

    bound: float

    def __init__(self, bound: float = 5.0):
        self.bound = bound


class Normal(Distribution):
    """
    Multivariate standard normal distribution.

    Note this distribution has infinite support, so it is not recommended that
    you use it with the spline coupling layers, which have compact support.
    """

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(stats.norm.logpdf(x), axis=-1)

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.normal(rngkey, shape=(nsamples, self.dim))


class TruncatedNormal(BoundedDistribution):
    """
    Truncated multivariate standard normal distribution.

    Along each dimension, it has support [-bound, bound].
    """

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(stats.truncnorm.logpdf(x, -self.bound, self.bound), axis=-1)

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.truncated_normal(
            rngkey, -self.bound, self.bound, shape=(nsamples, self.dim)
        )


class Beta(BoundedDistribution):
    """
    Shifted and scaled multivariate Beta distribution.

    Along each dimension, it has support [-bound, bound].

    The peakness parameter can be used to interpolate between a uniform and a normal
    distribution.

    It can be used as an alternative to the truncated normal.
    """

    peakness: float

    def __init__(self, bound: float = 5, peakness: float = 10):
        if peakness < 1:
            raise ValueError("peakness must be at least 1")
        self.peakness = peakness
        super().__init__(bound)

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(
            stats.beta.logpdf(
                x, self.peakness, self.peakness, loc=-self.bound, scale=2 * self.bound
            ),
            axis=-1,
        )

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return self.bound * (
            2
            * random.beta(
                rngkey,
                self.peakness,
                self.peakness,
                shape=(nsamples, self.dim),
            )
            - 1
        )


class Uniform(BoundedDistribution):
    """
    Multivariate uniform distribution.

    Along each dimension, it has support [-bound, bound].
    """

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(
            stats.uniform.logpdf(x, loc=-self.bound, scale=2 * self.bound), axis=-1
        )

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.uniform(
            rngkey,
            shape=(nsamples, self.dim),
            minval=-self.bound,
            maxval=self.bound,
        )
