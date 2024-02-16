"""Define the bijectors used in the normalizing flows."""

from functools import update_wrapper
from typing import Callable, Tuple

import jax.numpy as jnp
from jax import random

from ..typing import Pytree, Bijector_Info


class ForwardFunction:
    """
    Return the output and log_det of the forward bijection on the inputs.

    ForwardFunction of a Bijector, originally returned by the
    InitFunction of the Bijector.

    Parameters
    ----------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    inputs : jnp.ndarray
        The data to be transformed by the bijection.

    Returns
    -------
    outputs : jnp.ndarray
        Result of the forward bijection applied to the inputs.
    log_det : jnp.ndarray
        The log determinant of the Jacobian evaluated at the inputs.

    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self, params: Pytree, inputs: jnp.ndarray, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._func(params, inputs, **kwargs)


class InverseFunction:
    """
    Return the output and log_det of the inverse bijection on the inputs.

    InverseFunction of a Bijector, originally returned by the
    InitFunction of the Bijector.

    Parameters
    ----------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    inputs : jnp.ndarray
        The data to be transformed by the bijection.

    Returns
    -------
    outputs : jnp.ndarray
        Result of the inverse bijection applied to the inputs.
    log_det : jnp.ndarray
        The log determinant of the Jacobian evaluated at the inputs.

    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self, params: Pytree, inputs: jnp.ndarray, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._func(params, inputs, **kwargs)


class InitFunction:
    """
    Initialize the corresponding Bijector.

    InitFunction returned by the initialization of a Bijector.

    Parameters
    ----------
    rng : jnp.ndarray
        A Random Number Key from jax.random.PRNGKey.
    input_dim : int
        The input dimension of the bijection.

    Returns
    -------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    forward_fun : ForwardFunction
        The forward function of the Bijector.
    inverse_fun : InverseFunction
        The inverse function of the Bijector.

    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self, rng: jnp.ndarray, input_dim: int, **kwargs
    ) -> Tuple[Pytree, ForwardFunction, InverseFunction]:
        return self._func(rng, input_dim, **kwargs)


class Bijector:
    """Wrapper class for bijector functions."""

    def __init__(self, func: Callable) -> None:
        self._func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> Tuple[InitFunction, Bijector_Info]:
        return self._func(*args, **kwargs)


@Bijector
def InvSoftplus(
    column_idx: int, sharpness: float = 1
) -> Tuple[InitFunction, Bijector_Info]:
    """
    Bijector that applies inverse softplus to the specified column(s).

    Applying the inverse softplus ensures that samples from that column will
    always be non-negative. This is because samples are the output of the
    inverse bijection -- so samples will have a softplus applied to them.

    Parameters
    ----------
    column_idx : int
        An index or iterable of indices corresponding to the column(s)
        you wish to be transformed.
    sharpness : float; default=1
        The sharpness(es) of the softplus transformation. If more than one
        is provided, the list of sharpnesses must be of the same length as
        column_idx.

    Returns
    -------
    InitFunction
        The InitFunction of the Softplus Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    """
    idx = jnp.atleast_1d(column_idx)
    k = jnp.atleast_1d(sharpness)
    if len(idx) != len(k) and len(k) != 1:
        raise ValueError(
            "Please provide either a single sharpness or one for each column index."
        )

    bijector_info = ("InvSoftplus", (column_idx, sharpness))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs.at[:, idx].set(
                jnp.log(-1 + jnp.exp(k * inputs[:, idx])) / k,
            )
            log_det = jnp.log(1 + jnp.exp(-k * outputs[:, idx])).sum(axis=1)
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs.at[:, idx].set(
                jnp.log(1 + jnp.exp(k * inputs[:, idx])) / k,
            )
            log_det = -jnp.log(1 + jnp.exp(-k * inputs[:, idx])).sum(axis=1)
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Reverse() -> Tuple[InitFunction, Bijector_Info]:
    """
    Create bijector that reverses the order of inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the the Reverse Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    """
    bijector_info = ("Reverse", ())

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs[:, ::-1]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs[:, ::-1]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Shuffle() -> Tuple[InitFunction, Bijector_Info]:
    """
    Create bijector that randomly permutes inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the Shuffle Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    """
    bijector_info = ("Shuffle", ())

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        perm = random.permutation(rng, jnp.arange(input_dim))
        inv_perm = jnp.argsort(perm)

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs[:, perm]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs[:, inv_perm]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def StandardScaler(
    means: jnp.array, stds: jnp.array
) -> Tuple[InitFunction, Bijector_Info]:
    """
    Create bijector that applies standard scaling to each input.

    Each input dimension i has an associated mean u_i and standard dev s_i.
    Each input is rescaled as (input[i] - u_i)/s_i, so that each input dimension
    has mean zero and unit variance.

    Parameters
    ----------
    means : jnp.ndarray
        The mean of each column.
    stds : jnp.ndarray
        The standard deviation of each column.

    Returns
    -------
    InitFunction
        The InitFunction of the StandardScaler Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    """
    bijector_info = ("StandardScaler", (means, stds))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = (inputs - means) / stds
            log_det = jnp.log(1 / jnp.prod(stds)) * jnp.ones(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs * stds + means
            log_det = jnp.log(jnp.prod(stds)) * jnp.ones(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def UniformDequantizer(column_idx: int) -> Tuple[InitFunction, Bijector_Info]:
    """
    Create bijector that dequantizes discrete variables with uniform noise.

    Dequantizers are necessary for modeling discrete values with a flow.
    Note that this isn't technically a bijector.

    Parameters
    ----------
    column_idx : int
        An index or iterable of indices corresponding to the column(s) with
        discrete values.

    Returns
    -------
    InitFunction
        The InitFunction of the UniformDequantizer Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    """
    bijector_info = ("UniformDequantizer", (column_idx,))
    column_idx = jnp.array(column_idx)

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            u = random.uniform(random.PRNGKey(0), shape=inputs[:, column_idx].shape)
            outputs = inputs.astype(float)
            outputs.at[:, column_idx].set(outputs[:, column_idx] + u)
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs.at[:, column_idx].set(jnp.floor(inputs[:, column_idx]))
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info
