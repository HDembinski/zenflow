"""Utility functions used in other modules."""

from typing import Tuple
from jaxtyping import Array
import jax.numpy as jnp
from jax.nn import softmax

__all__ = [
    "squareplus",
    "normalize_spline_params",
    "rational_quadratic_spline_forward",
    "rational_quadratic_spline_inverse",
]


def squareplus(x: Array, b: float = 4) -> Array:
    """Compute softplus-like activation."""
    return 0.5 * (x + jnp.sqrt(jnp.square(x) + b))


def normalize_spline_params(
    dx: Array, dy: Array, sl: Array
) -> Tuple[Array, Array, Array]:
    """
    Return normalised spline parameters.

    Parameters
    ----------
    dx : Array
        Step size parameters along x with range [-oo, oo].
    dy : Array
        Step size parameters along y with range [-oo, oo].
    sl : Array
        Slope parameters with range [-oo, oo].

    Returns
    -------
    dx, dy, sl
        Arrays with normalised parameters. Step sizes are positive and sum up to 1.
        Slope parameters are in range [0, oo].

    """
    dx = softmax(dx)
    dy = softmax(dy)
    sl = squareplus(sl)
    return dx, dy, sl


def rational_quadratic_spline_forward(
    x: Array, dx: Array, dy: Array, slope: Array
) -> Tuple[Array, Array]:
    """
    Apply rational quadratic spline to inputs and return outputs with log_det.

    This uses the piecewise rational quadratic spline developed in [1].

    Parameters
    ----------
    x : Array of shape (M, N)
        The inputs to be transformed. The inputs are transformed in the interval [0, 1].
        Values outside of the interval are returned unchanged.
    dx : Array of shape (M, N, K)
        The widths of the spline bins. The values must be positive and sum to unity.
    dy : Array of shape (M, N, K)
        The heights of the spline bins. The values must be positive and sum to unity.
    slope : Array of shape (M, N, K - 1)
        The derivatives at the inner spline knots. The values must be in the interval
        [0, oo].

    Returns
    -------
    y : Array of shape (M, N)
        The result of applying the splines to the inputs.
    log_det : Array of shape (M, N)
        The log determinant of the Jacobian at the inputs.

    References
    ----------
    [1] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios.
        Neural Spline Flows. arXiv:1906.04032, 2019.
        https://arxiv.org/abs/1906.04032
    [2] Rezende, Danilo Jimenez et al.
        Normalizing Flows on Tori and Spheres. arxiv:2002.02428, 2020
        http://arxiv.org/abs/2002.02428

    """
    (
        xk,
        yk,
        dxk,
        dyk,
        dk,
        dkp1,
        sk,
        out_of_bounds,
    ) = _compute_rqs_input(x, dx, dy, slope, True)

    # [1] Appendix A.1
    # calculate spline
    relx = (x - xk) / dxk
    num = dyk * (sk * relx**2 + dk * relx * (1 - relx))
    den = sk + (dkp1 + dk - 2 * sk) * relx * (1 - relx)
    y = yk + num / den

    # [1] Appendix A.2
    # calculate the log determinant
    dnum = dkp1 * relx**2 + 2 * sk * relx * (1 - relx) + dk * (1 - relx) ** 2
    dden = sk + (dkp1 + dk - 2 * sk) * relx * (1 - relx)
    log_det = 2 * jnp.log(sk) + jnp.log(dnum) - 2 * jnp.log(dden)

    # replace log_det for out-of-bounds values = 0
    log_det = jnp.where(out_of_bounds, 0, log_det)
    log_det = log_det.sum(axis=1)

    # replace out-of-bounds values with original values
    y = jnp.where(out_of_bounds, x, y)
    return y, log_det


def rational_quadratic_spline_inverse(
    y: Array, dx: Array, dy: Array, slope: Array
) -> Tuple[Array, Array]:
    """
    Apply the inverse rational quadratic spline mapping.

    This uses the piecewise rational quadratic spline developed in [1].

    Parameters
    ----------
    y : Array of shape (M, N)
        The inputs to be transformed. The inputs are transformed in the interval [0, 1].
        Values outside of the interval are returned unchanged.
    dx : Array of shape (M, N, K)
        The widths of the spline bins. The values must be positive and sum to unity.
    dy : Array of shape (M, N, K)
        The heights of the spline bins. The values must be positive and sum to unity.
    slope : Array of shape (M, N, K - 1)
        The derivatives at the inner spline knots. The values must be in the interval
        [0, oo].

    Returns
    -------
    x : Array of shape (M, N)
        The result of applying the inverse splines to the inputs.

    References
    ----------
    [1] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios.
        Neural Spline Flows. arXiv:1906.04032, 2019.
        https://arxiv.org/abs/1906.04032
    [2] Rezende, Danilo Jimenez et al.
        Normalizing Flows on Tori and Spheres. arxiv:2002.02428, 2020
        http://arxiv.org/abs/2002.02428

    """
    (
        xk,
        yk,
        dxk,
        dyk,
        dk,
        dkp1,
        sk,
        out_of_bounds,
    ) = _compute_rqs_input(y, dx, dy, slope, False)

    # [1] Appendix A.3
    # quadratic formula coefficients
    a = (dyk) * (sk - dk) + (y - yk) * (dkp1 + dk - 2 * sk)
    b = (dyk) * dk - (y - yk) * (dkp1 + dk - 2 * sk)
    c = -sk * (y - yk)

    relx = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
    x = relx * dxk + xk

    # replace out-of-bounds values with original values
    x = jnp.where(out_of_bounds, y, x)
    return x


def _compute_rqs_input(
    x: Array, dx: Array, dy: Array, slope: Array, forward: bool
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    # knot x-positions
    xk = _knots(dx)
    # knot y-positions
    yk = _knots(dy)
    # knot derivatives with boundary condition
    dk = jnp.pad(
        slope,
        [(0, 0)] * (len(slope.shape) - 1) + [(1, 1)],
        mode="constant",
        constant_values=1,
    )
    # knot slopes
    sk = dy / dx

    idx, out_of_bounds = _index(x, xk if forward else yk)

    # return spline parameters for the bin corresponding to each input
    return (
        jnp.take_along_axis(xk, idx, -1)[..., 0],
        jnp.take_along_axis(yk, idx, -1)[..., 0],
        jnp.take_along_axis(dx, idx, -1)[..., 0],
        jnp.take_along_axis(dy, idx, -1)[..., 0],
        jnp.take_along_axis(dk, idx, -1)[..., 0],
        jnp.take_along_axis(dk, idx + 1, -1)[..., 0],
        jnp.take_along_axis(sk, idx, -1)[..., 0],
        out_of_bounds,
    )


def _knots(dx):
    return jnp.pad(
        jnp.cumsum(dx, axis=-1),
        [(0, 0)] * (len(dx.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=0,
    )


def _index(x, xk):
    out_of_bounds = (x < 0) | (x >= 1)
    idx = jnp.sum(xk <= x[..., None], axis=-1)[..., None] - 1
    # if x is out of bounds, we can return any valid index value,
    # since the results are discarded in the end
    idx = jnp.clip(idx, 0, xk.shape[-1] - 1)
    return idx, out_of_bounds
