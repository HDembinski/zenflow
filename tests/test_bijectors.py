from neural_flow import bijectors as bi
import jax
from jax import numpy as jnp
from numpy.testing import assert_allclose

KEY = jax.random.PRNGKey(0)


def test_ShiftBounds():
    x = jnp.array([[1, 5], [3, 4], [6, 2]])
    sb = bi.ShiftBounds()
    variables = sb.init(KEY, x, None)
    (z, log_det), updates = sb.apply(
        variables, x, None, train=True, mutable=["batch_stats"]
    )
    assert jnp.all(z.max(axis=0) <= 4)
    assert jnp.all(z.min(axis=0) >= -4)
    assert_allclose(updates["batch_stats"]["xmin"], [1, 2])
    assert_allclose(updates["batch_stats"]["xmax"], [6, 5])
    x2 = sb.apply(updates, z, None, method="inverse")
    assert_allclose(x2, x)


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
    (z, log_det), updates = chain.apply(
        variables, x, None, train=True, mutable=["batch_stats"]
    )
    assert_allclose(z, [[-4, 0, -4], [0, -4, 0], [4, 4, 4]])
    assert_allclose(log_det, jnp.full(3, 2.94), atol=1e-2)
    x2 = chain.apply(updates, z, None, method="inverse")
    assert_allclose(x2, x)


def test_NeuralSplineCoupling():
    x = jnp.array([[1.5, 2], [1, 3.5], [3.5, 4]])
    c = jnp.array([[1], [2], [3]])
    nsc = bi.NeuralSplineCoupling()
    variables = nsc.init(KEY, x, c)
    (y, log_det), updates = nsc.apply(
        variables, x, c, train=True, mutable=["batch_stats"]
    )
    assert_allclose(y, x)


def test_RollingSplineCoupling():
    x = jnp.array([[1.5, 2], [1, 3.5], [4.5, 6]])
    c = jnp.array([[1], [2], [3]])
    rsc = bi.RollingSplineCoupling(layers=(4,))
    variables = rsc.init(KEY, x, c)
    print(variables)
    (y, log_det), updates = rsc.apply(
        variables, x, c, train=True, mutable=["batch_stats"]
    )
    assert_allclose(y, [[1.5, 2], [1, 3.5], [4.5, 6]])
