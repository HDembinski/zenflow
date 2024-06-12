"""The Flow class which implements a trainable conditional normalizing flow."""

from typing import Union, Optional
from flax.typing import Array

import jax.numpy as jnp
import jax

from .distributions import Distribution, Beta
from .bijectors import Bijector, Chain
from flax import linen as nn

__all__ = ["Flow"]


class Flow(nn.Module):
    """A conditional normalizing flow."""

    bijector: Bijector
    latent: Distribution = Beta()

    def __call__(
        self,
        x: Array,
        c: Optional[Array] = None,
        *,
        train: bool = False,
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
        x, log_det = self.bijector(x, _normalize_c(c), train)
        log_prob = self.latent.log_prob(x) + log_det
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
            c = None
        else:
            size = conditions_or_size.shape[0]
            c = _normalize_c(conditions_or_size)
        x = self.latent.sample(size, jax.random.PRNGKey(seed))
        x = self.bijector.inverse(x, c)
        return x

    def _steps(self, x, c: Optional[Array] = None, *, inverse: bool = False):
        if not isinstance(self.bijector, Chain):
            raise ValueError("only for Chain bijector")

        c = _normalize_c(c)

        results = []
        if inverse:
            for bijector in self.bijector[::-1]:
                x = bijector.inverse(x, c)
                results.append(x)
        else:
            for bijector in self.bijector:
                x, _ = bijector(x, c, False)
                results.append(x)
        return results


def _normalize_c(c: Optional[Array]):
    if c is not None and c.ndim == 1:
        c = c.reshape(-1, 1)
    return c
