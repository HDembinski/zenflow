from zenflow import bijectors as bi
import jax
from jax import numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from typing import Tuple
from flax.typing import Array

KEY = jax.random.PRNGKey(0)


def test_Bijector():
    with pytest.raises(TypeError):
        bi.Bijector()

    class Foo(bi.Bijector):
        def __call__(
            self, x: Array, c: Array, train: bool = False
        ) -> Tuple[Array, Array]:
            return super().__call__(x, c, train)

        def inverse(self, x: Array, c: Array) -> Array:
            return super().inverse(x, c)

    foo = Foo()

    x = jnp.array((1, 2, 3))
    with pytest.raises(NotImplementedError):
        foo(x, x)
    with pytest.raises(NotImplementedError):
        foo.inverse(x, x)


def test_ShiftBounds_1():
    x = jnp.array([[1, 5], [3, 4], [6, 2]])
    sb = bi.ShiftBounds()
    variables = sb.init(KEY, x, None)
    (y, log_det), updates = sb.apply(
        variables, x, None, train=True, mutable=["batch_stats"]
    )
    assert_allclose(updates["batch_stats"]["xmin"], [1, 2])
    assert_allclose(updates["batch_stats"]["xmax"], [6, 5])

    z_ref = (x - x.min(0)) / (x.max(0) - x.min(0))
    y_ref = 0.99 * z_ref + (1 - z_ref) * 0.01

    assert_allclose(y, y_ref, atol=5e-6)

    x2 = sb.apply(updates, y, None, method="inverse")
    assert_allclose(x2, x, atol=1e-6)


def test_ShiftBounds_2():
    jacobi = pytest.importorskip("jacobi")

    x = jnp.column_stack(
        [
            2 * jax.random.uniform(KEY, shape=(10,)) - 1,
            jax.random.normal(KEY, shape=(10,)) * 10 - 10,
        ]
    )

    tr = bi.ShiftBounds()
    variables = tr.init(KEY, x, x)
    (y, log_det), variables = tr.apply(
        variables, x, x, mutable="batch_stats", train=True
    )

    # eliminate the extreme values from the data set, jacobi does not work for those
    mask = ~(
        (y[:, 0] == tr.margin)
        | (y[:, 1] == tr.margin)
        | (y[:, 0] == 1 - tr.margin)
        | (y[:, 1] == 1 - tr.margin)
    )
    x = x[mask]
    log_det = tr.apply(variables, x, x)[1]

    def fn(x):
        v = jnp.array([x])
        y = tr.apply(variables, v, v)[0]
        return y.reshape(-1)

    jac = [jacobi.jacobi(fn, xi)[0] for xi in x]
    log_det_ref = np.log(np.abs([np.linalg.det(jaci) for jaci in jac]))
    assert_allclose(log_det, log_det_ref, rtol=2e-2)


def test_Roll():
    x = jnp.array([[1, 5], [3, 4], [6, 2]])
    roll = bi.Roll()
    variables = roll.init(KEY, x, None)
    (z, log_det) = roll.apply(variables, x, None, train=True)
    assert_allclose(z, jnp.array([[5, 1], [4, 3], [2, 6]]))
    assert_allclose(log_det, jnp.zeros(3))
    x2 = roll.apply(variables, z, None, method="inverse")
    assert_allclose(x2, x)


def test_Chain_1():
    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    chain = bi.Chain([bi.Roll(), bi.Roll()])
    assert len(chain) == 2
    variables = chain.init(KEY, x, None)
    (z, log_det) = chain.apply(variables, x, None, train=True)
    assert_allclose(z, [[2, 3, 1], [5, 6, 4]])
    assert_allclose(log_det, jnp.zeros(2))
    x2 = chain.apply(variables, z, None, method="inverse")
    assert_allclose(x2, x)


def test_Chain_2():
    x = jnp.array([[2.5, 2, 3], [1, 3.5, 4.5], [4, 5, 6]])
    chain = bi.Chain([bi.ShiftBounds(), bi.Roll()])
    variables = chain.init(KEY, x, None)
    (y, log_det), updates = chain.apply(
        variables, x, None, train=True, mutable=["batch_stats"]
    )
    assert_allclose(y, [[0.01, 0.5, 0.01], [0.5, 0.01, 0.5], [0.99, 0.99, 0.99]])

    log_det_ref = chain[0].apply(
        {"batch_stats": updates["batch_stats"]["bijectors_0"]}, x, None
    )[1]

    assert_allclose(log_det, log_det_ref, atol=5e-6)
    x2 = chain.apply(updates, y, None, method="inverse")
    assert_allclose(x2, x)


def test_Chain_3():
    x = jnp.array([[1.5, 2], [1, 3.5], [3.5, 4]])
    c = jnp.array([[1.0], [2.0], [3.0]])
    chain = bi.chain(
        bi.ShiftBounds(),
        bi.NeuralSplineCoupling(),
        bi.Roll(),
        bi.NeuralSplineCoupling(),
    )
    variables = chain.init(KEY, x, c)
    (y, log_det), updates = chain.apply(
        variables, x, c, train=True, mutable=["batch_stats"]
    )
    variables = {"params": variables["params"], "batch_stats": updates["batch_stats"]}
    (y, log_det) = chain.apply(variables, x, c, train=False)
    x2 = chain.apply(variables, y, c, method="inverse")
    assert_allclose(x2, x, rtol=1e-5)


def test_NeuralSplineCoupling_1():
    x = jnp.array([[1.5, 2], [1, 3.5], [3.5, 4]])
    c = jnp.array([[1.0], [2.0], [3.0]])
    nsc = bi.NeuralSplineCoupling()
    variables = nsc.init(KEY, x, c)
    (y, log_det) = nsc.apply(variables, x, c, train=False)
    x2 = nsc.apply(variables, y, c, method="inverse")
    assert_allclose(x2, x, atol=1e-5)


def test_NeuralSplineCoupling_2():
    x = jnp.array([[1.5, 2, 3.3], [1, 3.5, 4.5], [3.5, 4, 5.5]])
    xt, xc = bi.NeuralSplineCoupling._split(x)
    assert xt.shape[1] == 1
    assert xc.shape[1] == 2


def test_rolling_spline_coupling():
    x = jnp.array([[1.5, 2], [1, 3.5], [3.5, 4]])
    c = jnp.array([[1.0], [2.0], [3.0]])
    rsc = bi.rolling_spline_coupling(x.shape[1], layers=(64, 64))
    variables = rsc.init(KEY, x, c)
    (y, log_det), updates = rsc.apply(
        variables, x, c, train=True, mutable=["batch_stats"]
    )
    variables = {"params": variables["params"], "batch_stats": updates["batch_stats"]}
    (y, log_det) = rsc.apply(variables, x, c, train=False)
    x2 = rsc.apply(variables, y, c, method="inverse")
    assert_allclose(x2, x, atol=1e-4)


def test_rolling_spline_coupling_bad_input():

    with pytest.raises(ValueError):
        bi.rolling_spline_coupling(0)

    with pytest.raises(ValueError):
        bi.rolling_spline_coupling(1)

    with pytest.raises(ValueError):
        bi.rolling_spline_coupling(2, margin=-0.1)

    with pytest.raises(ValueError):
        bi.rolling_spline_coupling(2, margin=0.51)


def test_TransformToBound_1():
    x = jnp.column_stack(
        [
            2 * jax.random.uniform(KEY, shape=(100,)) - 1,
            jax.random.normal(KEY, shape=(100,)) * 10 - 10,
            jax.random.exponential(KEY, shape=(100,)) * 10 + 10,
            1 - jax.random.exponential(KEY, shape=(100,)),
        ]
    )

    bounds = [
        (-1, 1),
        (-jnp.inf, jnp.inf),
        (10, jnp.inf),
        (-jnp.inf, 1),
    ]

    tr = bi.TransformToBound(bounds, momentum=0)
    variables = tr.init(KEY, x, x)
    (y, log_det), variables = tr.apply(
        variables, x, x, mutable="batch_stats", train=True
    )
    x2 = tr.apply(variables, y, y, method="inverse")

    assert y.shape == x.shape
    assert x2.shape == x.shape

    ymin = jnp.min(y, axis=0)
    ymax = jnp.max(y, axis=0)
    assert_array_equal(ymin > 0, True)
    assert_array_equal(ymax < 1, True)
    assert_allclose(ymin, 0, atol=5e-2)
    assert_allclose(ymax, 1, atol=5e-2)

    assert_allclose(x, x2, atol=1e-4)


def test_TransformToBound_2():
    jacobi = pytest.importorskip("jacobi")

    x = jnp.column_stack(
        [
            2 * jax.random.uniform(KEY, shape=(10,)) - 1,
            jax.random.normal(KEY, shape=(10,)) * 10 - 10,
            jax.random.exponential(KEY, shape=(10,)) * 10 + 10,
            1 - jax.random.exponential(KEY, shape=(10,)),
        ]
    )

    bounds = [
        (-1, 1),
        (-jnp.inf, jnp.inf),
        (10, jnp.inf),
        (-jnp.inf, 1),
    ]

    tr = bi.TransformToBound(bounds, momentum=0)
    variables = tr.init(KEY, x, x)
    (y, log_det), variables = tr.apply(
        variables, x, x, mutable="batch_stats", train=True
    )

    def fn(x):
        v = jnp.array([x])
        y = tr.apply(variables, v, v)[0]
        return y.reshape(-1)

    jac = [jacobi.jacobi(fn, xi)[0] for xi in x]
    log_det_ref = np.log(np.abs([np.linalg.det(jaci) for jaci in jac]))
    assert_allclose(log_det, log_det_ref, atol=1e-3)
