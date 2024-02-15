"""Define the bijectors used in the normalizing flows."""

from typing import Tuple
from .typing import Array, Pytree
import abc
import jax
from jax import numpy as jnp
from .utils import RationalQuadraticSpline, DenseReluNetwork
from jax.nn import softmax, softplus


class Bijector(abc.ABC):
    """Bijector base class."""

    def init(self, rngkey: Array, x: Array, c: Array):
        pass

    @abc.abstractmethod
    def forward(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        return NotImplemented

    @abc.abstractmethod
    def inverse(self, params: Pytree, x: Array, c: Array) -> Tuple[Array, Array]:
        return NotImplemented


class ShiftBounds(Bijector):
    """Shifts all values."""

    def __init__(self, bound: float = 5):
        self.bound = 5

    def init(self, rngkey: Array, x: Array, c: Array):
        xmin = x.min(axis=0)
        xmax = x.max(axis=0)
        self.xmean = (xmin + xmax) / 2
        self.xscale = (xmax - xmin) / 2
        self.xscale /= self.bound

    def forward(self, params, x: Array, c: Array):
        y = (x - self.xmean) / self.xscale
        log_det = jnp.log(jnp.prod(1 / self.xscale)) * jnp.ones(x.shape[0])
        return y, log_det

    def inverse(self, params, x: Array, c: Array):
        y = x * self.xscale + self.xmean
        log_det = jnp.log(jnp.prod(self.xscale)) * jnp.ones(x.shape[0])
        return y, log_det


class Roll(Bijector):
    """Roll inputs along their last column."""

    def __init__(self, shift: int = 1):
        self.shift = shift

    def forward(self, params: Pytree, x: Array, c: Array):
        x = jnp.roll(x, shift=self.shift, axis=-1)
        log_det = jnp.zeros(x.shape[0])
        return x, log_det

    def inverse(self, params: Pytree, x: Array, c: Array):
        x = jnp.roll(x, shift=-self.shift, axis=-1)
        log_det = jnp.zeros(x.shape[0])
        return x, log_det


class NeuralSplineCoupling(Bijector):
    """Coupling layer bijection with rational quadratic splines."""

    def __init__(
        self,
        K: int = 16,
        B: float = 5,
        hidden_layers: int = 2,
        hidden_dim: int = 128,
        periodic: bool = False,
    ):
        self.K = K
        self.B = B
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.periodic = periodic

    def init(self, rngkey, x: Array, c: Array):
        self.input_dim = x.shape[1]
        self.n_conditions = c.shape[1]

        # variables that determine NN self.params
        self.upper_dim = self.input_dim // 2
        # variables self.transformed by the NN
        self.lower_dim = self.input_dim - self.upper_dim

        # create the neural network that will take in the upper dimensions and
        # will return the spline parameters to transform the lower dimensions
        self.network = DenseReluNetwork(
            (3 * self.K - 1 + int(self.periodic)) * self.lower_dim,
            self.hidden_layers,
            self.hidden_dim,
        )
        return self.network.init(
            rngkey, jnp.zeros((1, self.upper_dim + self.n_conditions))
        )

    # calculate spline parameters as a function of the upper variables
    def spline_params(self, params, upper, conditions):
        x = jnp.hstack((upper, conditions))[:, : self.upper_dim + self.n_conditions]
        outputs = self.network.apply(params, x)
        outputs = jnp.reshape(
            outputs, [-1, self.lower_dim, 3 * self.K - 1 + int(self.periodic)]
        )
        W, H, D = jnp.split(outputs, [self.K, 2 * self.K], axis=2)
        W = 2 * self.B * softmax(W)
        H = 2 * self.B * softmax(H)
        D = softplus(D)
        return W, H, D

    def _transform(self, params, x, c, inverse):
        # lower dimensions are transformed as function of upper dimensions
        upper, lower = x[:, : self.upper_dim], x[:, self.upper_dim :]
        # widths, heights, derivatives = function(upper dimensions)
        W, H, D = self.spline_params(params, upper, c)
        # transform the lower dimensions with the Rational Quadratic Spline
        lower, log_det = RationalQuadraticSpline(
            lower, W, H, D, self.B, self.periodic, inverse=inverse
        )
        y = jnp.hstack((upper, lower))
        return y, log_det

    def forward(self, params, x, c):
        return self._transform(params, x, c, False)

    def inverse(self, params, x, c):
        return self._transform(params, x, c, True)


class Chain(Bijector):
    """Chain of other bjiectors."""

    def __init__(self, *bijectors):
        self.bijectors = bijectors

    def init(self, rngkey: Array, x: Array, c: Array):
        params = []
        for bijector in self.bijectors:
            subkey, rngkey = jax.random.split(rngkey)
            param = bijector.init(subkey, x, c)
            params.append(param)
        return params

    def forward(self, params: Pytree, x: Array, c: Array):
        log_dets = jnp.zeros(x.shape[0])
        for bijector, param in zip(self.bijectors, params):
            x, log_det = bijector.forward(param, x, c)
            log_dets += log_det
        return x, log_dets

    def inverse(self, params: Pytree, x: Array, c: Array):
        log_dets = jnp.zeros(x.shape[0])
        for bijector, param in zip(self.bijectors[::-1], params[::-1]):
            x, log_det = bijector.inverse(param, x, c)
            log_dets += log_det
        return x, log_dets


class RollingSplineCoupling(Chain):
    """Alternates between NeuralSplineCoupling and Roll."""

    def __init__(
        self,
        nlayers: int,
        shift: int = 1,
        K: int = 16,
        B: float = 5,
        hidden_layers: int = 2,
        hidden_dim: int = 128,
        periodic: bool = False,
    ):
        super().__init__(
            *(
                NeuralSplineCoupling(
                    K=K,
                    B=B,
                    hidden_layers=hidden_layers,
                    hidden_dim=hidden_dim,
                    periodic=periodic,
                ),
                Roll(shift),
            )
            * nlayers
        )
