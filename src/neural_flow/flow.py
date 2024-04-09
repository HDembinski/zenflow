"""The Flow class which implements a trainable conditional normalizing flow."""

from typing import Union, Optional
from jaxtyping import Array

import jax.numpy as jnp
import jax

from .distributions import Distribution, Beta
from .bijectors import Bijector
from flax import linen as nn

__all__ = ["Flow"]


class Flow(nn.Module):
    """A conditional normalizing flow."""

    bijector: Bijector
    latent: Distribution = Beta()

    @nn.compact
    def __call__(
        self, x: Array, c: Optional[Array] = None, *, train: bool = False
    ) -> Array:
        """
        Return log-likelihood of the samples.

        Parameters
        ----------
        x : Array of shape (N, D)
            N samples from a D-dimensional distribution. It is not necessary to
            normalize this distribution or to transform it to look gaussian, but doing
            so might accelerate convergence.
        c : Array of shape (N, K) or None
            N values from a K-dimensional vector of variables which determines the shape
            of the D-dimensional distribution.
        train : bool, optional (default = False)
            Whether to run in training mode (update BatchNorm statistics, etc.).

        """
        if c is None:
            c = jnp.zeros((x.shape[0], 0))
        elif c.ndim == 1:
            c = c.reshape(-1, 1)
        u, log_det = self.bijector(x, c, train)
        log_prob = self.latent.log_prob(u) + log_det
        log_prob = jnp.nan_to_num(log_prob, nan=-jnp.inf)
        return log_prob

    def sample(
        self,
        conditions_or_size: Union[Array, int],
        *,
        seed: int = 0,
    ) -> Array:
        """
        Return samples from the learned distribution.

        Parameters
        ----------
        conditions_or_size: Array of shape (N, K) or int
            If the distribution depends on a vector of conditional variables, you need
            to pass one vector here for each random sample that should be generated. If
            the distribution does not depend on conditional variables, you can directly
            pass the number of random samples here that should be generated.
        seed: int (default = 0)
            Seed to use for generating samples.

        """
        if isinstance(conditions_or_size, int):
            size = conditions_or_size
            c = jnp.zeros((size, 0))
        else:
            size = conditions_or_size.shape[0]
            c = conditions_or_size
            if c.ndim == 1:
                c = c.reshape(-1, 1)
        u = self.latent.sample(size, jax.random.PRNGKey(seed))
        x = self.bijector.inverse(u, c)
        return x
