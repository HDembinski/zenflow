"""Define the bijectors used in the normalizing flows."""

from typing import Tuple, Callable
from .typing import Array, Pytree
from abc import ABC, abstractmethod
import jax
from jax import numpy as jnp
from .utils import rational_quadratic_spline, FeedForwardNetwork
from jax.nn import softmax, softplus
from flax import linen as nn


class Bijector(ABC):
    """Bijector base class."""

    def init(self, rngkey: Array, x: Array, c: Array) -> Pytree:
        return ()

    @abstractmethod
    def forward(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        return NotImplemented


class ShiftBounds(Bijector):
    """Shifts all values."""

    def __init__(self, bound: float = 4.0):
        self.bound = bound

    def init(self, rngkey: Array, x: Array, c: Array) -> Pytree:
        xmin = x.min(axis=0)
        xmax = x.max(axis=0)
        self.xmean = (xmax + xmin) / 2
        self.xscale = 2 * self.bound / (xmax - xmin)
        return ()

    def forward(self, params, x: Array, c: Array) -> Tuple[Array, Array]:
        y = (x - self.xmean) * self.xscale
        log_det = jnp.sum(jnp.log(self.xscale)) * jnp.ones(x.shape[0])
        return y, log_det

    def inverse(self, params, x: Array, c: Array) -> Tuple[Array, Array]:
        y = x / self.xscale + self.xmean
        log_det = -jnp.sum(jnp.log(self.xscale)) * jnp.ones(x.shape[0])
        return y, log_det


class Roll(Bijector):
    """Roll inputs along their last column."""

    def __init__(self, shift: int = 1):
        self.shift = shift

    def forward(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        x = jnp.roll(x, shift=self.shift, axis=-1)
        log_det = jnp.zeros(x.shape[0])
        return x, log_det

    def inverse(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        x = jnp.roll(x, shift=-self.shift, axis=-1)
        log_det = jnp.zeros(x.shape[0])
        return x, log_det


class NeuralSplineCoupling(Bijector):
    """Coupling layer bijection with rational quadratic splines."""

    def __init__(
        self,
        knots: int = 16,
        bound: float = 5,
        periodic: bool = False,
        make_network: Callable[[int], nn.Module] = lambda out_dim: FeedForwardNetwork(
            out_dim, 2, 128
        ),
    ):
        self.knots = knots
        self.bound = bound
        self.periodic = periodic
        self.make_network = make_network

    def init(self, rngkey: Array, x: Array, c: Array) -> Pytree:
        self.input_dim = x.shape[1]
        self.n_conditions = c.shape[1]

        # variables that determine NN self.params
        self.upper_dim = self.input_dim // 2
        # variables self.transformed by the NN
        self.lower_dim = self.input_dim - self.upper_dim

        # create the neural network that will take in the upper dimensions and
        # will return the spline parameters to transform the lower dimensions
        self.network = self.make_network(
            (3 * self.knots - 1 + int(self.periodic)) * self.lower_dim
        )
        return self.network.init(
            rngkey, jnp.zeros((1, self.upper_dim + self.n_conditions))
        )

    # calculate spline parameters as a function of the upper variables
    def spline_params(
        self, params: Pytree, x: Array, c: Array
    ) -> Tuple[Array, Array, Array]:
        xc = jnp.hstack((x, c))[:, : self.upper_dim + self.n_conditions]
        outputs = self.network.apply(params, xc)
        outputs = jnp.reshape(
            outputs, [-1, self.lower_dim, 3 * self.knots - 1 + int(self.periodic)]
        )
        W, H, D = jnp.split(outputs, [self.knots, 2 * self.knots], axis=2)
        W = 2 * self.bound * softmax(W)
        H = 2 * self.bound * softmax(H)
        D = softplus(D)
        return W, H, D

    def _transform(
        self, params: Pytree, x: Array, c: Array, inverse: bool
    ) -> Tuple[Array, Array]:
        # lower dimensions are transformed as function of upper dimensions
        upper, lower = x[:, : self.upper_dim], x[:, self.upper_dim :]
        # widths, heights, derivatives = function(upper dimensions)
        W, H, D = self.spline_params(params, upper, c)
        # transform the lower dimensions with the Rational Quadratic Spline
        lower, log_det = rational_quadratic_spline(
            lower, W, H, D, self.bound, self.periodic, inverse=inverse
        )
        y = jnp.hstack((upper, lower))
        return y, log_det

    def forward(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        return self._transform(params, x, c, False)

    def inverse(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        return self._transform(params, x, c, True)


class Chain(Bijector):
    """Chain of other bjiectors."""

    def __init__(self, *bijectors: Bijector):
        self.bijectors = bijectors

    def init(self, rngkey: Array, x: Array, c: Array) -> Pytree:
        params = []
        for bijector in self.bijectors:
            rngkey, subkey = jax.random.split(rngkey)
            param = bijector.init(subkey, x, c)
            params.append(param)
        return params

    def forward(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        log_dets = jnp.zeros(x.shape[0])
        for bijector, param in zip(self.bijectors, params):
            x, log_det = bijector.forward(param, x, c)
            log_dets += log_det
        return x, log_dets

    def inverse(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        log_dets = jnp.zeros(x.shape[0])
        for bijector, param in zip(self.bijectors[::-1], params[::-1]):
            x, log_det = bijector.inverse(param, x, c)
            log_dets += log_det
        return x, log_dets


class RollingSplineCoupling(Chain):
    """Alternates between NeuralSplineCoupling and Roll."""

    def __init__(
        self,
        knots: int = 16,
        bound: float = 5,
        periodic: bool = False,
        make_network: Callable[[int], nn.Module] = lambda out_dim: FeedForwardNetwork(
            out_dim, 2, 128
        ),
    ):
        self.knotsnots = knots
        self.bound = bound
        self.periodic = periodic
        self.make_network = make_network

    def init(self, rngkey: Array, x: Array, c: Array) -> Pytree:
        input_dim = x.shape[1]
        bijectors = []
        for j in range(input_dim):
            bijectors.append(
                NeuralSplineCoupling(
                    self.knotsnots, self.bound, self.periodic, self.make_network
                )
            )
            bijectors.append(Roll())
        super().__init__(*bijectors)
        return super().init(rngkey, x, c)
