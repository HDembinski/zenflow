from numpy.testing import assert_allclose
from neural_flow import utils
import pytest
import numpy as np


def test_rational_quadratic_spline_1():
    x = np.linspace(-1, 2, 10).reshape(-1, 1)
    W = np.tile([0.25, 0.25, 0.25, 0.25], len(x)).reshape(*x.shape, -1)
    H = np.tile([0.25, 0.25, 0.25, 0.25], len(x)).reshape(*x.shape, -1)
    D = np.tile([1.0, 1.0, 1.0], len(x)).reshape(*x.shape, -1)
    y, log_det = utils.rational_quadratic_spline_forward(x, W, H, D)
    assert_allclose(y, x)


def test_rational_quadratic_spline_2():
    jacobi = pytest.importorskip("jacobi")

    rng = np.random.default_rng(1)

    x = np.linspace(-0.1, 1.1, 1000).reshape(1000, 1)

    scale = 0.1
    knots = 3
    dx, dy, slope = utils.normalize_spline_params(
        scale * rng.normal(size=knots),
        scale * rng.normal(size=knots),
        scale * rng.normal(size=knots - 1),
    )

    nx = np.prod(x.shape)
    dx = np.tile(dx, nx).reshape(*x.shape, -1)
    dy = np.tile(dy, nx).reshape(*x.shape, -1)
    slope = np.tile(slope, nx).reshape(*x.shape, -1)

    y, log_det = utils.rational_quadratic_spline_forward(x, dx, dy, slope)

    j, je = jacobi.jacobi(
        lambda x: utils.rational_quadratic_spline_forward(
            x.reshape(1000, 1), dx, dy, slope
        )[0].reshape(-1),
        x.reshape(-1),
        diagonal=True,
    )

    assert_allclose(y, x, atol=0.1)
    assert_allclose(log_det, np.log(j), atol=0.01)

    x2 = utils.rational_quadratic_spline_inverse(y, dx, dy, slope)
    assert_allclose(x2, x, atol=1e-5)


def test_index():
    x = np.array([-2, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5]).reshape(1, -1)
    xk = np.array([-1, 0, 1]).reshape(1, 3)
    expected = []
    for xi in x[0]:
        if xi < xk[0][0]:
            expected.append(0)
        elif xk[0][-1] <= xi:
            expected.append(2)
        else:
            for j in range(len(xk[0]) - 1):
                if xk[0][j] <= xi < xk[0][j + 1]:
                    expected.append(j)
                    break
    ind, oob = utils._index(x, xk)
    assert_allclose(ind[0, :, 0], expected)


def test_knots():
    dx = np.array((0.25, 0.25, 0.25))
    xk = utils._knots(dx)
    assert_allclose(xk, [0, 0.25, 0.5, 0.75])
