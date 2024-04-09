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

    def __repr__(self):
        """Return string representation."""
        return f"""{self.__class__.__name__}()"""


class BoundedDistribution(Distribution):
    """
    Base class for bounded distributions.

    Along each dimension, it has support [-bound, bound].
    """

    bound: float

    def __init__(self, bound: float = 5.0):
        self.bound = bound

    def __repr__(self):
        """Return string representation."""
        return f"""{self.__class__.__name__}(bound={self.bound})"""


class Normal(Distribution):
    """
    Multivariate standard normal distribution.

    Warning: This distribution has infinite support. It is not recommended that
    you use it with the spline coupling layers. Use TruncatedNormal or Beta
    instead.
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

    It is a drop-in alternative to TruncatedNormal. In contrast to the former, the
    probability is exactly zero at the boundary. The peakness parameter can be used to
    interpolate between a uniform and a normal distribution. The default value is chosen
    so that the variance of the shifted and scaled Beta distribution is equal to a
    standard normal distribution.
    """

    peakness: float

    def __init__(self, bound: float = 5.0, peakness: float = 12.0):
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

    def __repr__(self):
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"bound={self.bound}, "
            f"peakness={self.peakness})"
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
