"""Define the bijectors used in the normalizing flows."""

from typing import Tuple, Optional, Sequence, Callable
from jaxtyping import Array
from abc import ABC, abstractmethod
from jax import numpy as jnp
from .utils import rational_quadratic_spline
from flax import linen as nn
import numpy as np


__all__ = [
    "Bijector",
    "ShiftBounds",
    "Roll",
    "NeuralSplineCoupling",
    "Chain",
    "chain",
    "rolling_spline_coupling",
]


class Bijector(nn.Module, ABC):
    """Bijector base class."""

    @abstractmethod
    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array, c: Array) -> Array:
        return NotImplemented


class Chain(Bijector):
    """Chain of other bjiectors."""

    bijectors: Sequence[Bijector]

    @nn.compact
    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        log_det = jnp.zeros(x.shape[0])
        for bijector in self.bijectors:
            x, ld = bijector(x, c, train)
            log_det += ld
        return x, log_det

    def inverse(self, x: Array, c: Array) -> Array:
        for bijector in self.bijectors[::-1]:
            x = bijector.inverse(x, c)
        return x


def chain(*bijectors):
    """Create chain from bijector arguments."""
    return Chain(bijectors)


class ShiftBounds(Bijector):
    """Shifts all values."""

    bound: float = 4.0

    @nn.nowrap
    def _compute_mean_scale(self, xmin, xmax):
        xmean = (xmax + xmin) / 2
        xscale = 2 * self.bound / (xmax - xmin)
        return xmean, xscale

    @nn.compact
    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        ra_min = self.variable(
            "batch_stats", "xmin", lambda s: jnp.full(s, np.inf), x.shape[1]
        )
        ra_max = self.variable(
            "batch_stats", "xmax", lambda s: jnp.full(s, -np.inf), x.shape[1]
        )

        if train:
            xmin = jnp.minimum(ra_min.value, x.min(axis=0))
            xmax = jnp.maximum(ra_max.value, x.max(axis=0))
            if not self.is_initializing():
                ra_min.value = xmin
                ra_max.value = xmax
        else:
            xmin = ra_min.value
            xmax = ra_max.value

        xmean, xscale = self._compute_mean_scale(xmin, xmax)

        y = (x - xmean) * xscale
        log_det = jnp.sum(jnp.log(xscale)) * jnp.ones(x.shape[0])
        return y, log_det

    def inverse(self, x: Array, c: Array) -> Array:
        xmin = self.get_variable("batch_stats", "xmin")
        xmax = self.get_variable("batch_stats", "xmax")
        xmean, xscale = self._compute_mean_scale(xmin, xmax)

        y = x / xscale + xmean
        return y


class Roll(Bijector):
    """Roll inputs along their last column."""

    shift: int = 1

    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        x = jnp.roll(x, shift=self.shift, axis=-1)
        log_det = jnp.zeros(x.shape[0])
        return x, log_det

    def inverse(self, x: Array, c: Array) -> Array:
        x = jnp.roll(x, shift=-self.shift, axis=-1)
        return x


class NeuralSplineCoupling(Bijector):
    """Coupling layer bijection with rational quadratic splines."""

    knots: int = 16
    bound: float = 5
    layers: Sequence[int] = (128, 128)
    act: Callable = nn.swish

    @nn.nowrap
    @staticmethod
    def _split(x: Array):
        x_dim = x.shape[1]
        x_split = x_dim // 2
        assert x_split > 0 and x_split < x_dim
        lower = x[:, :x_split]
        upper = x[:, x_split:]
        return lower, upper

    @nn.compact
    def _spline_params(
        self, lower: Array, upper: Array, c: Array, train: bool
    ) -> Tuple[Array, Array, Array]:
        # calculate spline parameters as a function of the upper variables
        dim = lower.shape[1]
        spline_dim = 3 * self.knots - 1
        x = jnp.hstack((upper, c))

        # feed forward network
        x = nn.BatchNorm(use_running_average=not train)(x)
        for width in self.layers:
            x = nn.Dense(width)(x)
            x = self.act(x)
        x = nn.Dense(dim * spline_dim)(x)
        x = jnp.reshape(x, [-1, dim, spline_dim])

        W, H, D = jnp.split(x, [self.knots, 2 * self.knots], axis=2)
        W = 2 * self.bound * nn.softmax(W)
        H = 2 * self.bound * nn.softmax(H)
        D = nn.softplus(D)
        return W, H, D

    def _transform(
        self, x: Array, c: Array, inverse: bool, train: bool
    ) -> Tuple[Array, Optional[Array]]:
        lower, upper = self._split(x)
        W, H, D = self._spline_params(lower, upper, c, train)
        lower, log_det = rational_quadratic_spline(
            lower, W, H, D, self.bound, self.periodic, inverse
        )
        y = jnp.hstack((lower, upper))
        return y, log_det

    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        return self._transform(x, c, False, train)

    def inverse(self, x: Array, c: Array) -> Array:
        return self._transform(x, c, True, False)[0]


def rolling_spline_coupling(
    dim: int,
    knots: int = 16,
    layers: Sequence[int] = (128, 128),
):
    """
    Create a rolling spline coupling chain of bijectors.

    The chain starts with ShiftBounds and then alternates between
    NeuralSplineCoupling and Roll once for each dimension in the input.
    The input must be at least two-dimensional.
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    bijectors = [ShiftBounds()]
    for _ in range(dim - 1):
        bijectors.append(NeuralSplineCoupling(knots=knots, layers=layers))
        bijectors.append(Roll())
    bijectors.append(NeuralSplineCoupling(knots=knots, layers=layers))
    # we can skip last Roll which is superfluous
    return Chain(bijectors)
