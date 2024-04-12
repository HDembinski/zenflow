from neural_flow import bijectors as bi
import jax
from jax import numpy as jnp
from numpy.testing import assert_allclose

KEY = jax.random.PRNGKey(0)


def test_ShiftBounds():
    x = jnp.array([[1, 5], [3, 4], [6, 2]])
    sb = bi.ShiftBounds()
    variables = sb.init(KEY, x, None)
    (y, log_det), updates = sb.apply(
        variables, x, None, train=True, mutable=["batch_stats"]
    )
    assert_allclose(updates["batch_stats"]["xmin"], [1, 2])
    assert_allclose(updates["batch_stats"]["xmax"], [6, 5])

    z_ref = (x - x.min(0)) / (x.max(0) - x.min(0))
    y_ref = 0.9 * z_ref + (1 - z_ref) * 0.1

    assert_allclose(y, y_ref, atol=5e-6)

    x2 = sb.apply(updates, y, None, method="inverse")
    assert_allclose(x2, x, atol=1e-6)


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
    assert_allclose(y, [[0.1, 0.5, 0.1], [0.5, 0.1, 0.5], [0.9, 0.9, 0.9]])

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
    t, c = bi.NeuralSplineCoupling._split(x)
    assert t.shape[1] == 1
    assert c.shape[1] == 0


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
    assert_allclose(x2, x, atol=1e-5)
