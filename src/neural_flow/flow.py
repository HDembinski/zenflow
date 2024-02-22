"""Define the Flow object that defines the normalizing flow."""

from typing import Any, Union, Optional, Tuple
from .typing import Pytree, Array

import jax.numpy as jnp
import jax

from .distributions import Distribution, Uniform
from .bijectors import (
    Bijector,
    Chain,
    RollingSplineCoupling,
    ShiftBounds,
)

__all__ = ["Flow"]


class Flow:
    """A conditional normalizing flow."""

    latent: Distribution
    bijector: Bijector

    def __init__(
        self,
        *bijectors: Bijector,
        latent: Distribution = Uniform(5),
    ) -> None:
        self.bijector = (
            Chain(ShiftBounds(), RollingSplineCoupling())
            if not bijectors
            else bijectors[0] if len(bijectors) == 1 else Chain(*bijectors)
        )
        self.latent = latent

    @staticmethod
    def _normalize(x: Array, c: Optional[Array]) -> Tuple[Array, Array]:
        if jnp.ndim(x) < 2:
            x = x.reshape(-1, 1)
        if c is None:
            c = jnp.zeros((x.shape[0], 0))
        elif jnp.ndim(c) < 2:
            c = c.reshape(-1, 1)
        return x, c

    def init(
        self,
        rngkey: Any,
        x: Array,
        c: Optional[Array],
    ) -> Pytree:
        x, c = self._normalize(x, c)
        self.c_dim = c.shape[1]
        self._c_mean = jnp.mean(c, axis=0)
        self._c_scale = 1 / jnp.std(c, axis=0)
        self.latent.init(x)
        return self.bijector.init(rngkey, x, c)

    def log_prob(self, params: Pytree, x: Array, c: Optional[Array], **kwargs) -> Tuple[Array, Pytree]:
        """
        Calculate log probability density of inputs.

        Parameters
        ----------
        params: Pytree
            Bijector parameters.
        x : Array
            Input data for which log probability density is calculated.
        c : Array
            Conditional data for the bijectors.

        Returns
        -------
        Array
            Device array of shape (inputs.shape[0],).

        """
        x, c = self._normalize(x, c)
        u, log_det, updates = self.bijector.forward(params, x, c, **kwargs)
        log_prob = self.latent.log_prob(u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = jnp.nan_to_num(log_prob, nan=-jnp.inf)
        return log_prob, updates

    def sample(
        self,
        params: Pytree,
        conditions_or_size: Union[Array, int],
        seed: int = 0,
    ) -> Array:
        if isinstance(conditions_or_size, int):
            if self.c_dim > 0:
                raise ValueError("Second argument must be an array of conditions")
            size = conditions_or_size
            c = jnp.zeros((size, 0))
        else:
            if self.c_dim == 0:
                raise ValueError("Second argument must be number of samples")
            size = conditions_or_size.shape[0]
        if jnp.ndim(c) < 2:
            c = c.reshape(-1, 1)
        if c.shape[1] != self.c_dim:
            msg = f"Number of conditions must be {self.c_dim}, got {c.shape[1]}"
            raise ValueError(msg)
        u = self.latent.sample(size, jax.random.PRNGKey(seed))
        x = self.bijector.inverse(params, u, c)
        return x
