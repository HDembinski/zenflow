"""Train flow."""

from .flow import Flow
import jax.numpy as jnp
from typing import Callable
import numpy as np
import jax
import optax


def train(
    flow: Flow,
    X_train: jnp.ndarray,
    C_train: jnp.ndarray,
    X_test: jnp.ndarray,
    C_test: jnp.ndarray,
    epochs: int = 100,
    batch_size: int = 1024,
    optimizer: Callable = None,
    patience: int = 10,
    seed: int = 0,
    progress: bool = True,
) -> list:
    """Trains the normalizing flow on the provided inputs."""
    root_key = jax.random.PRNGKey(seed)
    init_key, batch_key = jax.random.split(root_key)

    X_train = jax.device_put(X_train)
    X_test = jax.device_put(X_test)
    C_train = jax.device_put(C_train)
    C_test = jax.device_put(C_test)

    params = flow.init(init_key, X_train, C_train)

    tx = optax.adam(learning_rate=1e-3) if optimizer is None else optimizer
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, x, c):
        return -jnp.mean(flow.log_prob(params, x, c))

    @jax.jit
    def step(params, opt_state, x, c):
        gradients = jax.grad(loss_fn)(params, x, c)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    losses = [loss_fn(params, X_train, C_train).item()]
    test_losses = []

    if progress:
        from rich.progress import track

        loop = track(range(epochs))
    else:
        loop = range(epochs)

    best_epoch = 0
    best_params = params
    for epoch in loop:
        batch_key, permute_key = jax.random.split(batch_key)
        perm = jax.random.permutation(permute_key, X_train.shape[0])
        X_perm = X_train[perm]
        C_perm = C_train[perm]

        # loop through batches and step optimizer
        for batch_idx in range(0, len(X_perm), batch_size):
            X = X_perm[batch_idx : batch_idx + batch_size]
            C = C_perm[batch_idx : batch_idx + batch_size]
            params, opt_state = step(params, opt_state, X, C)

        losses.append(loss_fn(params, X, C).item())
        test_losses.append(loss_fn(params, X_test, C_test).item())

        if test_losses[-1] < test_losses[best_epoch]:
            best_epoch = epoch
            best_params = params

        stop = np.isnan(losses[-1]) or (
            len(test_losses) > 2 * patience
            and not np.min(test_losses[-patience:])
            < np.min(test_losses[-2 * patience : -patience])
        )

        if stop:
            break

    return best_params, best_epoch, losses, test_losses
