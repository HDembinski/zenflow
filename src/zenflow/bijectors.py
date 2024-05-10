"""Bijectors used in conditional normalizing flows."""

from typing import Tuple, Sequence, Callable, Union
from jaxtyping import Array
from abc import ABC, abstractmethod
from jax import numpy as jnp
from .utils import (
    normalize_spline_params,
    rational_quadratic_spline_forward,
    rational_quadratic_spline_inverse,
)
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
    """
    Bijector base class.

    A bijector is a basic element that defines the normalizing flow. The bijector is
    learned during training to transform a simple base distribution to the target
    distribution.
    """

    @abstractmethod
    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        """
        Transform samples from the target distribution to the base distribution.

        Parameters
        ----------
        x : Array of shape (N, D)
            N samples from a D-dimensional target distribution. It is not necessary to
            standardize it or transform it to look more gaussian, but doing so might
            accelerate convergence or allow one to use a simpler bijector.
        c : Array of shape (N, K) or None
            N values from a K-dimensional vector of variables which determines the shape
            of the D-dimensional distribution.
        train : bool, optional (default = False)
            Whether to run in training mode (update BatchNorm statistics, etc.).

        Returns
        -------
        y : Array of shape (N, D)
            N samples of the base distribution.
        log_det : Array of shape (N,)
            Logarithm of the determinant of the transformation.

        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x: Array, c: Array) -> Array:
        """
        Transform samples from the base distribution to the target distribution.

        The log-determinant is not returned in the inverse pass, since it is not needed.

        Parameters
        ----------
        x : Array of shape (N, D)
            N samples from the D-dimensional base distribution.
        c : Array of shape (N, K) or None
            N values from a K-dimensional vector of variables which determines the shape
            of the D-dimensional target distribution.

        Returns
        -------
        y : Array of shape (N, D)
            N samples of the target distribution.

        """
        raise NotImplementedError


class Chain(Bijector, Sequence):
    """
    Chain of other bjiectors.

    The forward transform calls bijectors in order and applies the forward transform of
    each and accumulates the log-determinants.

    The inverse transform calls the bijectors in reverse order and applies the inverse
    transform of each.
    """

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

    def __getitem__(self, idx: Union[int, slice]):
        """Get bijector at location idx."""
        return self.bijectors[idx]

    def __len__(self):
        """Return number of bijectors in the chain."""
        return len(self.bijectors)


def chain(*bijectors):
    """Create a chain directly from a variable number of bijector arguments."""
    return Chain(bijectors)


class ShiftBounds(Bijector):
    """
    Shift all values into the interval [margin, 1 - margin].

    This bijector keeps track of the smallest and largest inputs along each dimension of
    the target distribution and applies an affine transformation so that all values are
    inside a hypercube where each side starts at `margin` and ends at `1 - margin`.

    This transformation is necessary before applying the first NeuralSplineCoupling,
    which only transforms samples inside a hypercube with coordinates in the interval
    (0, 1) along each dimension. A value margin > 0 guarantees that values are never
    mapped to the boundary values, where the latent distribution may be exactly zero.
    """

    margin: float = 0.01

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

        xscale = 1 / (xmax - xmin)
        z = (x - xmin) * xscale
        # If test sample has more extreme values than train sample, it is possible to
        # get z values outside of the interval [0, 1], which may cause the latent
        # distribution to be evaluated outside of its non-zero domain. We clip the
        # values as a workaround.
        z = jnp.clip(z, 0, 1)
        y = (1 - self.margin) * z + (1 - z) * self.margin

        abs_deriv = (1 - 2 * self.margin) * xscale
        log_det = jnp.sum(jnp.log(abs_deriv)) * jnp.ones(x.shape[0])
        return y, log_det

    def inverse(self, y: Array, c: Array) -> Array:
        xmin = self.get_variable("batch_stats", "xmin")
        xmax = self.get_variable("batch_stats", "xmax")

        z = (y - self.margin) / (1 - 2 * self.margin)
        x = xmax * z + (1 - z) * xmin

        return x


class Roll(Bijector):
    """
    Roll inputs along their last column.

    This bijector should be used together with a NeuralSplineCoupling. Couplings use the
    upper dimensions of the input sample and the conditional variables to transform the
    lower dimensions of the input sample. Roll mixes the upper and lower dimensions. One
    should apply at least D-1 Rolls for D dimensional input to transform all dimensions.
    """

    shift: int = 1

    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        x = jnp.roll(x, shift=self.shift, axis=-1)
        log_det = jnp.zeros(x.shape[0])
        return x, log_det

    def inverse(self, x: Array, c: Array) -> Array:
        x = jnp.roll(x, shift=-self.shift, axis=-1)
        return x


class NeuralSplineCoupling(Bijector):
    """
    Coupling layer with transforms with rational quadratic splines.

    This coupling transform uses a rational quadratic spline, which is analytically
    invertible. Couplings use the upper dimensions of the input sample and the
    conditional variables to transform the lower dimensions of the input sample.

    The spline only transform values in a hypercube with side intervals [0, 1]. For
    values outside of the hypercube the identity transform is applied.

    For a derivation, discussion, and more information, see:

    Durkan, C., Bekasov, A., Murray, I., and Papamakarios, G. (2019). “Neural Spline
    Flows,” In: Advances in Neural Information Processing Systems, pp. 7509–7520.
    """

    knots: int = 16
    layers: Sequence[int] = (128, 128)
    act: Callable[[Array], Array] = nn.swish

    @nn.nowrap
    @staticmethod
    def _split(x: Array):
        x_dim = x.shape[1]
        x_split = x_dim // 2
        assert x_split > 0 and x_split < x_dim
        return x[:, :x_split], x[:, x_split:]

    @nn.compact
    def _spline_params(
        self, x: Array, c: Array, train: bool
    ) -> Tuple[Array, Array, Array, Array, Array]:
        # xt are transformed conditionally based on values xc
        xt, xc = self._split(x)

        dim = xt.shape[1]
        spline_dim = 3 * self.knots - 1

        # calculate spline parameters as a function of xc variables
        # and external conditional variables c
        x = jnp.hstack((xc, c))
        x = nn.BatchNorm(use_running_average=not train)(x)
        for width in self.layers:
            x = nn.Dense(width)(x)
            x = self.act(x)
        x = nn.Dense(dim * spline_dim)(x)
        x = x.reshape((xt.shape[0], dim, spline_dim))

        return (
            xt,
            xc,
            *normalize_spline_params(
                x[..., : self.knots],
                x[..., self.knots : 2 * self.knots],
                x[..., 2 * self.knots :],
            ),
        )

    def __call__(self, x: Array, c: Array, train: bool = False) -> Tuple[Array, Array]:
        xt, xc, dx, dy, sl = self._spline_params(x, c, train)
        yt, log_det = rational_quadratic_spline_forward(xt, dx, dy, sl)
        y = jnp.hstack((yt, xc))
        return y, log_det

    def inverse(self, y: Array, c: Array) -> Array:
        yt, yc, dx, dy, sl = self._spline_params(y, c, False)
        xt = rational_quadratic_spline_inverse(yt, dx, dy, sl)
        x = jnp.hstack((xt, yc))
        return x


def rolling_spline_coupling(
    dim: int, knots: int = 16, layers: Sequence[int] = (128, 128), margin: float = 0.1
) -> Chain:
    """
    Create a chain of rolling spline couplings.

    The chain starts with ShiftBounds and then alternates between
    NeuralSplineCoupling and Roll once for each dimension in the input.
    The input must be at least two-dimensional for this to work.

    Parameters
    ----------
    dim: int
        The dimension of the target distribution.
    knots : int (default = 16)
        Number of knots used by the spline.
    layers: sequence of int (default = (128, 128))
        Sequence of neurons per hidden layer in the feed-forward network which computes
        the spline parameters from the upper dimensions of the input and the conditional
        variables.
    margin : float (default = 0.1)
        Safety margin for ShiftBounds. Must be in the interval [0, 0.5].

    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if margin < 0 or margin > 0.5:
        raise ValueError("margin must be in the interval [0, 0.5]")
    bijectors = [ShiftBounds(margin=margin)]
    for _ in range(dim - 1):
        bijectors.append(NeuralSplineCoupling(knots=knots, layers=layers))
        bijectors.append(Roll())
    bijectors.append(NeuralSplineCoupling(knots=knots, layers=layers))
    # we can skip last Roll, latent distribution is invariant to Roll
    return Chain(bijectors)
