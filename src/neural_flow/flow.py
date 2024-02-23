"""Define the Flow object that defines the normalizing flow."""

from typing import Union, Optional, Tuple
from .typing import Array

import jax.numpy as jnp
import jax

from .distributions import Distribution, Uniform
from .bijectors import (
    Bijector,
    Chain,
    RollingSplineCoupling,
    ShiftBounds,
)
from flax import linen as nn

__all__ = ["Flow"]


class Flow(nn.Module):
    """A conditional normalizing flow."""

    latent: Distribution = Uniform()
    bijector: Bijector = Chain([ShiftBounds(), RollingSplineCoupling()])

    @nn.compact
    def __call__(
        self, x: Array, c: Optional[Array], train: bool = False
    ) -> Tuple[Array]:
        u, log_det = self.bijector(x, c, train)
        log_prob = self.latent.log_prob(u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = jnp.nan_to_num(log_prob, nan=-jnp.inf)
        return log_prob

    def sample(
        self,
        conditions_or_size: Union[Array, int],
        seed: int = 0,
    ) -> Array:
        if isinstance(conditions_or_size, int):
            size = conditions_or_size
            c = jnp.zeros((size, 0))
        else:
            size = conditions_or_size.shape[0]
        if jnp.ndim(c) < 2:
            c = c.reshape(-1, 1)
        u = self.latent.sample(size, jax.random.PRNGKey(seed))
        x = self.bijector.inverse(u, c)
        return x
