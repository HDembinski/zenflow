from typing import Callable, Tuple

import jax.numpy as np
from jax import random
from jax.example_libraries.stax import Dense, LeakyRelu, serial

from pzflow import bijectors


def build_bijector_from_info(info):
    """Build a Bijector from a Bijector_Info object"""

    # recurse through chains
    if info[0] == "Chain":
        return bijectors.Chain(*(build_bijector_from_info(i) for i in info[1]))
    # build individual bijector from name and parameters
    else:
        return getattr(bijectors, info[0])(*info[1])


def DenseReluNetwork(
    out_dim: int, hidden_layers: int, hidden_dim: int
) -> Tuple[Callable, Callable]:
    """Create a dense neural network with Relu after hidden layers.

    Parameters
    ----------
    out_dim : int
        The output dimension.
    hidden_layers : int
        The number of hidden layers
    hidden_dim : int
        The dimension of the hidden layers

    Returns
    -------
    init_fun : function
        The function that initializes the network. Note that this is the
        init_function defined in the Jax stax module, which is different
        from the functions of my InitFunction class.
    forward_fun : function
        The function that passes the inputs through the neural network.
    """
    init_fun, forward_fun = serial(
        *(Dense(hidden_dim), LeakyRelu) * hidden_layers,
        Dense(out_dim),
    )
    return init_fun, forward_fun


def gaussian_error_model(
    key, X: np.ndarray, Xerr: np.ndarray, nsamples: int
) -> np.ndarray:
    """
    Default Gaussian error model were X are the means and Xerr are the stds.
    """

    eps = random.normal(key, shape=(X.shape[0], nsamples, X.shape[1]))

    return X[:, None, :] + eps * Xerr[:, None, :]


def RationalQuadraticSpline(
    inputs: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    B: float,
    periodic: bool = False,
    inverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rational quadratic spline to inputs and return outputs with log_det.

    Applies the piecewise rational quadratic spline developed in [1].

    Parameters
    ----------
    inputs : np.ndarray
        The inputs to be transformed.
    W : np.ndarray
        The widths of the spline bins.
    H : np.ndarray
        The heights of the spline bins.
    D : np.ndarray
        The derivatives of the inner spline knots.
    B : float
        Range of the splines.
        Outside of (-B,B), the transformation is just the identity.
    inverse : bool; default=False
        If True, perform the inverse transformation.
        Otherwise perform the forward transformation.
    periodic : bool; default=False
        Whether to make this a periodic, Circular Spline [2].

    Returns
    -------
    outputs : np.ndarray
        The result of applying the splines to the inputs.
    log_det : np.ndarray
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
    # knot x-positions
    xk = np.pad(
        -B + np.cumsum(W, axis=-1),
        [(0, 0)] * (len(W.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-B,
    )
    # knot y-positions
    yk = np.pad(
        -B + np.cumsum(H, axis=-1),
        [(0, 0)] * (len(H.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-B,
    )
    # knot derivatives
    if periodic:
        dk = np.pad(D, [(0, 0)] * (len(D.shape) - 1) + [(1, 0)], mode="wrap")
    else:
        dk = np.pad(
            D,
            [(0, 0)] * (len(D.shape) - 1) + [(1, 1)],
            mode="constant",
            constant_values=1,
        )
    # knot slopes
    sk = H / W

    # if not periodic, out-of-bounds inputs will have identity applied
    # if periodic, we map the input into the appropriate region inside
    # the period. For now, we will pretend all inputs are periodic.
    # This makes sure that out-of-bounds inputs don't cause problems
    # with the spline, but for the non-periodic case, we will replace
    # these with their original values at the end
    out_of_bounds = (inputs <= -B) | (inputs >= B)
    masked_inputs = np.where(out_of_bounds, np.abs(inputs) - B, inputs)

    # find bin for each input
    if inverse:
        idx = np.sum(yk <= masked_inputs[..., None], axis=-1)[..., None] - 1
    else:
        idx = np.sum(xk <= masked_inputs[..., None], axis=-1)[..., None] - 1

    # get kx, ky, kyp1, kd, kdp1, kw, ks for the bin corresponding to each input
    input_xk = np.take_along_axis(xk, idx, -1)[..., 0]
    input_yk = np.take_along_axis(yk, idx, -1)[..., 0]
    input_dk = np.take_along_axis(dk, idx, -1)[..., 0]
    input_dkp1 = np.take_along_axis(dk, idx + 1, -1)[..., 0]
    input_wk = np.take_along_axis(W, idx, -1)[..., 0]
    input_hk = np.take_along_axis(H, idx, -1)[..., 0]
    input_sk = np.take_along_axis(sk, idx, -1)[..., 0]

    if inverse:
        # [1] Appendix A.3
        # quadratic formula coefficients
        a = (input_hk) * (input_sk - input_dk) + (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        b = (input_hk) * input_dk - (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        c = -input_sk * (masked_inputs - input_yk)

        relx = 2 * c / (-b - np.sqrt(b**2 - 4 * a * c))
        outputs = relx * input_wk + input_xk
        # if not periodic, replace out-of-bounds values with original values
        if not periodic:
            outputs = np.where(out_of_bounds, inputs, outputs)

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx**2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (
            1 - relx
        )
        log_det = 2 * np.log(input_sk) + np.log(dnum) - 2 * np.log(dden)
        # if not periodic, replace log_det for out-of-bounds values = 0
        if not periodic:
            log_det = np.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

        return outputs, -log_det

    else:
        # [1] Appendix A.1
        # calculate spline
        relx = (masked_inputs - input_xk) / input_wk
        num = input_hk * (input_sk * relx**2 + input_dk * relx * (1 - relx))
        den = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (
            1 - relx
        )
        outputs = input_yk + num / den
        # if not periodic, replace out-of-bounds values with original values
        if not periodic:
            outputs = np.where(out_of_bounds, inputs, outputs)

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx**2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (
            1 - relx
        )
        log_det = 2 * np.log(input_sk) + np.log(dnum) - 2 * np.log(dden)
        # if not periodic, replace log_det for out-of-bounds values = 0
        if not periodic:
            log_det = np.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

        return outputs, log_det


def sub_diag_indices(
    inputs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for diagonal of 2D blocks in 3D array"""
    if inputs.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    nblocks = inputs.shape[0]
    ndiag = min(inputs.shape[1], inputs.shape[2])
    idx = (
        np.repeat(np.arange(nblocks), ndiag),
        np.tile(np.arange(ndiag), nblocks),
        np.tile(np.arange(ndiag), nblocks),
    )
    return idx
