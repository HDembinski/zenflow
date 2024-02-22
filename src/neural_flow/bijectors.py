"""Define the bijectors used in the normalizing flows."""

from typing import Tuple, Optional, Sequence
from .typing import Array
from abc import ABC, abstractmethod
from jax import numpy as jnp
from .utils import rational_quadratic_spline, SplineNetwork
from jax.nn import softmax, softplus
from flax import linen as nn
import numpy as np


__all__ = [
    "Bijector",
    "ShiftBounds",
    "Roll",
    "NeuralSplineCoupling",
    "Chain",
    "RollingSplineCoupling",
]


class Bijector(nn.Module, ABC):
    """Bijector base class."""

    @abstractmethod
    def __call__(self, x: Array, c: Array, train: bool) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array, c: Array) -> Array:
        return NotImplemented


class Chain(Bijector):
    """Chain of other bjiectors."""

    bijectors: Sequence[Bijector]

    @nn.compact
    def __call__(self, x: Array, c: Array, train: bool) -> Tuple[Array, Array]:
        log_det = jnp.zeros(x.shape[0])
        for bijector in self.bijectors:
            x, ld = bijector(x, c, train)
            log_det += ld
        return x, log_det

    def inverse(self, x: Array, c: Array) -> Array:
        for bijector in self.bijectors[::-1]:
            x = bijector.inverse(x, c)
        return x


class ShiftBounds(Bijector):
    """Shifts all values."""

    bound: float = 4.0

    @nn.nowrap
    def _compute_mean_scale(self, xmin, xmax):
        xmean = (xmax + xmin) / 2
        xscale = 2 * self.bound / (xmax - xmin)
        return xmean, xscale

    @nn.compact
    def __call__(self, x: Array, c: Array, train: bool) -> Tuple[Array, Array]:
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
        xmin = self.variable("batch_stats", "xmin").value
        xmax = self.variable("batch_stats", "xmax").value
        xmean, xscale = self._compute_mean_scale(xmin, xmax)

        y = x / xscale + xmean
        return y


class Roll(Bijector):
    """Roll inputs along their last column."""

    shift: int = 1

    def __call__(self, x: Array, c: Array, train: bool) -> Tuple[Array, Array]:
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
    periodic: bool = False
    layers: Sequence[int] = (128, 128)

    @nn.nowrap
    @staticmethod
    def _split(x: Array):
        x_dim = x.shape[1]
        x_split = x_dim // 2
        assert x_split > 0 and x_split < x_dim
        lower = x[:, :x_split]
        upper = x[:, x_split:]
        return lower, upper

    def _spline_params(
        self, lower: Array, upper: Array, c: Array, train: bool
    ) -> Tuple[Array, Array, Array]:
        # calculate spline parameters as a function of the upper variables
        dim = lower.shape[1]
        spline_dim = 3 * self.knots - 1 + int(self.periodic)
        x = jnp.hstack((upper, c))
        x = SplineNetwork(dim * spline_dim, self.layers)(x, train)
        x = jnp.reshape(x, [-1, dim, spline_dim])
        W, H, D = jnp.split(x, [self.knots, 2 * self.knots], axis=2)
        W = 2 * self.bound * softmax(W)
        H = 2 * self.bound * softmax(H)
        D = softplus(D)
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

    @nn.compact
    def __call__(self, x: Array, c: Array, train: bool) -> Tuple[Array, Array]:
        return self._transform(x, c, False, train)

    def inverse(self, x: Array, c: Array) -> Array:
        return self._transform(x, c, True, False)[0]


class RollingSplineCoupling(Bijector):
    """Alternates between NeuralSplineCoupling and Roll."""

    repeat: int = 1
    knots: int = 16
    bound: float = 5
    periodic: bool = False
    layers: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, x: Array, c: Array, train: bool) -> Tuple[Array, Array]:
        if self.is_initializing:
            self.bijections = []
            for _ in range(self.repeat):
                for _ in range(x.shape[1]):
                    self.bijections.append(
                        NeuralSplineCoupling(
                            self.knots, self.bound, self.periodic, self.layers
                        )
                    )
                    self.bijections.append(Roll())

        log_det = jnp.zeros(x.shape[0])
        for bi in self.bijections:
            x, ld = bi(x, c, train)
            log_det += ld
        return x, log_det

    def inverse(self, x: Array, c: Array) -> Array:
        for bi in self.bijections[::-1]:
            x = bi.inverse(x, c)
        return x
