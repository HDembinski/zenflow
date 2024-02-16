"""Define the Flow object that defines the normalizing flow."""

from typing import Any, Union

import jax.numpy as jnp
from jax import random

from .distributions import Distribution, Uniform
from .typing import Pytree, Array
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

    def init(
        self,
        rngkey: Any,
        x: Array,
        c: Array,
    ) -> Pytree:
        self.c_dim = c.shape[1]
        self._c_mean = jnp.mean(c, axis=0)
        self._c_scale = 1 / jnp.std(c, axis=0)
        self.latent.init(x)
        return self.bijector.init(rngkey, x, c)

    def _c_standardize(self, c: Array) -> Array:
        return (c - self._c_mean) * self._c_scale

    def log_prob(self, params: Pytree, x: Array, c: Array) -> Array:
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
        c = self._c_standardize(c)
        u, log_det = self.bijector.forward(params, x, c)
        log_prob = self.latent.log_prob(u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = jnp.nan_to_num(log_prob, nan=-jnp.inf)
        return log_prob

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
            c = self._c_standardize(conditions_or_size)
        if c.shape[1] != self.c_dim:
            msg = f"Number of conditions must be {self.c_dim}, got {c.shape[1]}"
            raise ValueError(msg)
        u = self.latent.sample(size, random.PRNGKey(seed))
        x = self.bijector.inverse(params, u, c)[0]
        return x
