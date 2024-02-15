"""Define the Flow object that defines the normalizing flow."""

from typing import Any, Union

import jax.numpy as jnp
from jax import random

from . import distributions
from .typing import Pytree
from .bijectors2 import (
    Bijector,
    Chain,
    RollingSplineCoupling,
    ShiftBounds,
)

__all__ = ["Flow"]


class Flow:
    """A conditional normalizing flow."""

    def __init__(
        self,
        bijector: Bijector = None,
        latent: distributions.LatentDist = None,
    ) -> None:
        self._bijector = (
            Chain(ShiftBounds(4.0), RollingSplineCoupling(2))
            if bijector is None
            else bijector
        )
        self._latent = distributions.Uniform(5) if latent is None else latent

    def init(
        self,
        rngkey: Any,
        x: jnp.ndarray,
        c: jnp.ndarray,
    ) -> Pytree:
        self._latent.init(x)
        self._c_mean = jnp.mean(c, axis=0)
        self._c_scale = 1 / jnp.std(c, axis=0)
        return self._bijector.init(rngkey, x, c)

    def _c_standardize(self, c: jnp.ndarray) -> jnp.ndarray:
        return (c - self._c_mean) * self._c_scale

    def log_prob(self, params: Pytree, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate log probability density of inputs.

        Parameters
        ----------
        params: Pytree
            Bijector parameters.
        x : jnp.ndarray
            Input data for which log probability density is calculated.
        c : jnp.ndarray
            Conditional data for the bijectors.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """
        c = self._c_standardize(c)
        u, log_det = self._bijector.forward(params, x, c)
        log_prob = self._latent.log_prob(u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = jnp.nan_to_num(log_prob, nan=-jnp.inf)
        return log_prob

    def sample(
        self,
        params: Pytree,
        conditions_or_samples: Union[jnp.ndarray, int],
        seed: int = 0,
    ) -> jnp.ndarray:
        if isinstance(conditions_or_samples, int):
            samples = conditions_or_samples
            c = jnp.zeros((samples, 0))
        else:
            samples = conditions_or_samples.shape[0]
            c = self._c_standardize(conditions_or_samples)
        # draw from latent distribution
        rngkey = random.PRNGKey(seed)
        u = self._latent.sample(samples, rngkey)
        # take the inverse back to the data distribution
        x = self._bijector.inverse(params, u, c)[0]
        return x
