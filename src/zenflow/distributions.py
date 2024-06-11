"""Base distributions used in conditional normalizing flows."""

from abc import ABC, abstractmethod
from flax.typing import Array
from typing import Optional
import jax.numpy as jnp
from jax import random
from jax.scipy import stats


class Distribution(ABC):
    """Distribution base class with infrastructure for lazy initialization."""

    __dim: Optional[int] = None

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
        if self.__dim is None:
            self.__dim = x.shape[-1]
        return self._log_prob_impl(x)

    @property
    def dim(self):
        return self.__dim

    @abstractmethod
    def _log_prob_impl(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, nsamples: int, rngkey: Array) -> Array: ...

    def __repr__(self):
        """Return string representation."""
        return f"""{self.__class__.__name__}()"""


class Normal(Distribution):
    """
    Multivariate normal distribution with mean 0.5 and standard deviation 0.1.

    Warning: This distribution has infinite support. It is not recommended to use it
    with spline coupling layers. Use :class:`TruncatedNormal` or :class:`Beta` instead.
    """

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(stats.norm.logpdf(x, loc=0.5, scale=0.1), axis=-1)

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return 0.5 + 0.1 * random.normal(rngkey, shape=(nsamples, self.dim))


class TruncatedNormal(Distribution):
    """
    Like :class:`Normal`, but truncated to the interval [0, 1].

    The probability to have samples exactly at the boundary is small, but non-zero.
    """

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(stats.truncnorm.logpdf(x, -5, 5, loc=0.5, scale=0.1), axis=-1)

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return 0.5 + 0.1 * random.truncated_normal(
            rngkey, -5, 5, shape=(nsamples, self.dim)
        )


class Beta(Distribution):
    """
    Multivariate beta distribution.

    It is a drop-in alternative to :class:`TruncatedNormal`. In contrast to the former,
    the probability is exactly zero at the boundary.

    The peakness parameter can be used to interpolate between a uniform and a normal
    distribution. The default value is chosen so that the variance is equal to
    :class:`Normal`.
    """

    peakness: float

    def __init__(self, peakness: float = 12.0):
        if peakness < 1:
            raise ValueError("peakness must be at least 1")
        self.peakness = peakness

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(
            stats.beta.logpdf(x, self.peakness, self.peakness),
            axis=-1,
        )

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.beta(
            rngkey,
            self.peakness,
            self.peakness,
            shape=(nsamples, self.dim),
        )

    def __repr__(self):
        """Return string representation."""
        return f"{self.__class__.__name__}(peakness={self.peakness})"


class Uniform(Distribution):
    """Multivariate uniform distribution."""

    def _log_prob_impl(self, x: Array) -> Array:
        return jnp.sum(stats.uniform.logpdf(x), axis=-1)

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.uniform(rngkey, shape=(nsamples, self.dim))
