"""Define utility functions for use in other modules."""

from typing import Tuple, Optional
from jaxtyping import Array
import jax.numpy as jnp


def _knots(dx, bound):
    return jnp.pad(
        -bound + jnp.cumsum(dx, axis=-1),
        [(0, 0)] * (len(dx.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-bound,
    )


def _index(x, xk, bound):
    out_of_bounds = (x < -bound) | (x >= bound)
    idx = jnp.sum(xk <= x[..., None], axis=-1)[..., None] - 1
    idx = jnp.clip(idx, 0, xk.shape[-1] - 1)
    return idx, out_of_bounds


def rational_quadratic_spline(
    inputs: Array, dx: Array, dy: Array, slope: Array, bound: float, inverse: bool
) -> Tuple[Array, Optional[Array]]:
    """
    Apply rational quadratic spline to inputs and return outputs with log_det.

    Applies the piecewise rational quadratic spline developed in [1].

    Parameters
    ----------
    inputs : jnp.ndarray
        The inputs to be transformed.
    dx : jnp.ndarray
        The widths of the spline bins.
    dy : jnp.ndarray
        The heights of the spline bins.
    slope : jnp.ndarray
        The derivatives of the inner spline knots.
    bound : float
        Range of the splines.
        Outside of (-B,B), the transformation is just the identity.
    inverse : bool
        If True, perform the inverse transformation.
        Otherwise perform the forward transformation.

    Returns
    -------
    outputs : jnp.ndarray
        The result of applying the splines to the inputs.
    log_det : jnp.ndarray or None
        The log determinant of the Jacobian at the inputs or None if
        if inverse=True.

    References
    ----------
    [1] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios.
        Neural Spline Flows. arXiv:1906.04032, 2019.
        https://arxiv.org/abs/1906.04032
    [2] Rezende, Danilo Jimenez et al.
        Normalizing Flows on Tori and Spheres. arxiv:2002.02428, 2020
        http://arxiv.org/abs/2002.02428

    """
    # knot x-positions
    xk = _knots(dx, bound)
    # knot y-positions
    yk = _knots(dy, bound)
    # knot derivatives with boundary condition
    dk = jnp.pad(
        slope,
        [(0, 0)] * (len(slope.shape) - 1) + [(1, 1)],
        mode="constant",
        constant_values=1,
    )
    # knot slopes
    sk = dy / dx

    idx, out_of_bounds = _index(inputs, yk if inverse else xk, bound)

    # get kx, ky, kyp1, kd, kdp1, kw, ks for the bin corresponding to each input
    input_xk = jnp.take_along_axis(xk, idx, -1)[..., 0]
    input_yk = jnp.take_along_axis(yk, idx, -1)[..., 0]
    input_dx = jnp.take_along_axis(dx, idx, -1)[..., 0]
    input_dy = jnp.take_along_axis(dy, idx, -1)[..., 0]
    input_dk = jnp.take_along_axis(dk, idx, -1)[..., 0]
    input_dkp1 = jnp.take_along_axis(dk, idx + 1, -1)[..., 0]
    input_sk = jnp.take_along_axis(sk, idx, -1)[..., 0]

    if inverse:
        # [1] Appendix A.3
        # quadratic formula coefficients
        a = (input_dy) * (input_sk - input_dk) + (inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        b = (input_dy) * input_dk - (inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        c = -input_sk * (inputs - input_yk)

        relx = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
        outputs = relx * input_dx + input_xk
        log_det = None
    else:
        # [1] Appendix A.1
        # calculate spline
        relx = (inputs - input_xk) / input_dx
        num = input_dy * (input_sk * relx**2 + input_dk * relx * (1 - relx))
        den = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        outputs = input_yk + num / den

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx**2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * jnp.log(input_sk) + jnp.log(dnum) - 2 * jnp.log(dden)
        # replace log_det for out-of-bounds values = 0
        log_det = jnp.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

    # replace out-of-bounds values with original values
    outputs = jnp.where(out_of_bounds, inputs, outputs)
    return outputs, log_det
