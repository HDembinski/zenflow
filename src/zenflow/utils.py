"""Utility functions used in other modules."""

from typing import Tuple
from flax.typing import Array
import jax.numpy as jnp

__all__ = [
    "squareplus",
    "normalize_spline_params",
    "rational_quadratic_spline_forward",
    "rational_quadratic_spline_inverse",
]


EPS = 1e-5


def squareplus(x: Array, b: float = 4) -> Array:
    """Compute softplus-like activation."""
    return 0.5 * (x + jnp.sqrt(jnp.square(x) + b))


def softmax_with_threshold(x: Array, threshold: float = 0) -> Array:
    """
    Similar softmax, but smallest possible value is threshold.

    Threshold must be positive and less then 1 / n. We use squareplus instead of
    exponential to obtain a softer gradient.
    """
    x = squareplus(x)
    n = x.shape[-1]
    c = threshold / (1 - n * threshold)
    xs = jnp.sum(x, axis=-1)[..., None]
    return (x / xs + c) / (1 + c * n)


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
    dx = softmax_with_threshold(dx, EPS)
    dy = softmax_with_threshold(dy, EPS)
    sl = squareplus(sl)
    return dx, dy, sl


def rational_quadratic_spline_forward(
    x: Array, dx: Array, dy: Array, slope: Array
) -> Tuple[Array, Array]:
    """
    Apply the spline transform to return outputs and log-determinant.

    This uses the piecewise rational quadratic spline developed in [1].

    To be efficient, this function is was written in a vectorization-friendly way. It
    computes the transform for a batch of M points, where each point has dimension N.
    Each dimension is transformed in parallel using an independent set of K spline
    variables for that dimension, where K is the number of knots of the spline. Since
    the transformation is done for each dimension independently, the Jacobian is
    triangular and the determinant is the product of its diagonal elements.

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
    log_det : Array of shape (M,)
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

    # [1] Appendix A.1, Eq. 19
    z = (x - xk) / dxk
    z = jnp.clip(z, EPS, 1 - EPS)
    az = 1 - z
    num = dyk * z * (sk * z + dk * az)
    den = sk + (dkp1 + dk - 2 * sk) * z * az
    y = yk + num / (den + EPS)

    # replace out-of-bounds values with original values
    y = jnp.where(out_of_bounds, x, y)

    # [1] Appendix A.2, Eq. 22
    num = z * (dkp1 * z + 2 * sk * az) + dk * az**2
    den = sk + (dkp1 + dk - 2 * sk) * z * az
    log_det = 2 * jnp.log(sk + EPS) + jnp.log(num + EPS) - 2 * jnp.log(den + EPS)

    # set log_det for out-of-bounds values to 0
    log_det = jnp.where(out_of_bounds, 0, log_det)
    log_det = log_det.sum(axis=1)

    return y, log_det


def rational_quadratic_spline_inverse(
    y: Array, dx: Array, dy: Array, slope: Array
) -> Tuple[Array, Array]:
    """
    Apply the inverse rational quadratic spline mapping.

    See rational_quadratic_spline_forward for implementation details.

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

    # [1] Appendix A.3, Eq. 29-32
    # quadratic formula coefficients
    a = dyk * (sk - dk) + (y - yk) * (dkp1 + dk - 2 * sk)
    b = dyk * dk - (y - yk) * (dkp1 + dk - 2 * sk)
    c = -sk * (y - yk)

    z = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
    x = z * dxk + xk

    # replace out-of-bounds values with original values
    x = jnp.where(out_of_bounds, y, x)
    return x


def _compute_rqs_input(
    x: Array, dx: Array, dy: Array, slope: Array, forward: bool
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    xk = _knots(dx)
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
