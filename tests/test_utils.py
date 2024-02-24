from neural_flow.utils import SplineNetwork
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose


def test_SplineNetwork():
    key = jax.random.PRNGKey(0)
    net = SplineNetwork(1, (4,))
    x = jnp.array([[1.0, 2.0]])

    variables = net.init(key, x)
    params = variables["params"]
    assert_allclose(params["Dense_0"]["kernel"], 0)
    assert_allclose(params["Dense_0"]["bias"], 0)
    assert_allclose(params["Dense_1"]["kernel"], 0)
    assert_allclose(params["Dense_1"]["bias"], 0)
    y = net.apply(variables, x)
    assert_allclose(y, 0)
