from neural_flow import Flow
import jax
import jax.numpy as jnp


def test_Flow():
    flow = Flow()
    x = jnp.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    variables = flow.init(jax.random.PRNGKey(0), x)
    log_prob, updates = flow.apply(variables, x, train=True, mutables=["batch_stats"])
    variables = {"params": variables["params"], "batch_stats": updates["batch_stats"]}
    flow.sample(variables)
