from zenflow import Flow
from zenflow.bijectors import ShiftBounds
import jax
import jax.numpy as jnp


def test_Flow_1():
    flow = Flow(ShiftBounds())
    x = jnp.array([[3.0, 2.0], [1.0, 4.0], [5.0, 6.0]])
    variables = flow.init(jax.random.PRNGKey(0), x)
    log_prob, variables = flow.apply(variables, x, train=True, mutable=["batch_stats"])
    x2 = flow.apply(variables, 1000, method="sample")
    assert x2.shape == (1000, 2)
    assert x[:, 0].min() >= 1
    assert x[:, 0].max() <= 5
    assert x[:, 1].min() >= 2
    assert x[:, 1].max() <= 6


def test_Flow_2():
    flow = Flow(ShiftBounds())
    x = jnp.array([[3.0, 2.0], [1.0, 4.0], [5.0, 6.0]])
    c = jnp.array([1.0, 2.0, 3.0])
    variables = flow.init(jax.random.PRNGKey(0), x)
    log_prob, variables = flow.apply(
        variables, x, c, train=True, mutable=["batch_stats"]
    )
    x2 = flow.apply(variables, c, method="sample")
    assert x2.shape == (3, 2)
