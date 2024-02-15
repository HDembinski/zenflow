"""Define the Flow object that defines the normalizing flow."""

from typing import Tuple, Any, Optional, Union

import jax.numpy as jnp
from jax import random

from . import distributions
from .typing import Pytree, Bijector_Info
from .bijectors import (
    Chain,
    InitFunction,
    RollingSplineCoupling,
    ShiftBounds,
)

__all__ = ["Flow"]


class Flow:
    """A conditional normalizing flow."""

    def __init__(
        self,
        latent: distributions.LatentDist = None,
    ) -> None:
        self._latent = distributions.Uniform(5) if latent is None else latent

    def init(
        self,
        X: jnp.ndarray,
        C: jnp.ndarray,
        rngkey: Any,
        bijector: Optional[Tuple[InitFunction, Bijector_Info]] = None,
    ) -> Pytree:
        self._latent.init(X)

        if bijector is None:
            bijector = Chain(
                ShiftBounds(X.min(axis=0), X.max(axis=0), 4.0),
                RollingSplineCoupling(X.shape[1], n_conditions=C.shape[1]),
            )

        self._c_mean = jnp.mean(C, axis=0)
        self._c_scale = 1 / jnp.std(C, axis=0)

        init_fun, self._bijector_info = bijector
        params, self._forward, self._inverse = init_fun(rngkey, X.shape[1])
        return params

    def _c_standardize(self, C: jnp.ndarray) -> jnp.ndarray:
        return (C - self._c_mean) * self._c_scale

    def log_prob(self, params: Pytree, X: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate log probability density of inputs.

        Parameters
        ----------
        params: Pytree
            Bijector parameters.
        X : jnp.ndarray
            Input data for which log probability density is calculated.
        C : jnp.ndarray
            Conditional data for the bijectors.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """
        c = self._c_standardize(C)
        u, log_det = self._forward(params, X, conditions=c)
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
        x = self._inverse(params, u, conditions=c)[0]
        return x