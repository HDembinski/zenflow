"""Define the latent distributions used in the normalizing flows."""

from abc import ABC, abstractmethod
from jaxtyping import Array
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal


class Distribution(ABC):
    """Base class for latent distributions."""

    def __init__(self):
        self.x_dim = 0
        self.is_initialized = False

    def lazy_init(self, x: Array):
        self.x_dim = x.shape[1]
        self.is_initialized = True
        return self.x_dim

    def log_prob(self, x: Array) -> Array:
        if not self.is_initialized:
            self.lazy_init(x)
        return self._log_prob_impl(x)

    @abstractmethod
    def _log_prob_impl(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, nsamples: int, rngkey: Array) -> Array: ...


class Normal(Distribution):
    """
    A multivariate Gaussian distribution with mean zero and unit variance.

    Note this distribution has infinite support, so it is not recommended that
    you use it with the spline coupling layers, which have compact support.
    If you do use the two together, you should set the support of the spline
    layers (using the spline parameter B) to be large enough that you rarely
    draw Gaussian samples outside the support of the splines.
    """

    def lazy_init(self, x: Array):
        dim = super().lazy_init(x)
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

    def __init__(self, bound: float = 5) -> None:
        self.bound = bound
        super().__init__()

    def _log_prob_impl(self, x: Array) -> Array:
        # which inputs are inside the support of the distribution
        mask = jnp.prod((x >= -self.bound) & (x <= self.bound), axis=-1)

        # calculate log_prob
        log_prob = jnp.where(
            mask,
            -self.x_dim * jnp.log(2 * self.bound),
            -jnp.inf,
        )

        return log_prob

    def sample(self, nsamples: int, rngkey: Array) -> Array:
        return random.uniform(
            rngkey,
            shape=(nsamples, self.x_dim),
            minval=-self.bound,
            maxval=self.bound,
        )
