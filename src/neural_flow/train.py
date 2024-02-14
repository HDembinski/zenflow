"""Train flow."""

from .flow import Flow
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from .typing import Bijector_Info
from .bijectors import InitFunction
import numpy as np
from jax import random, jit, value_and_grad
import optax


def train(
    flow: Flow,
    X_train: jnp.ndarray,
    C_train: jnp.ndarray,
    X_test: jnp.ndarray,
    C_test: jnp.ndarray,
    bijector: Optional[Tuple[InitFunction, Bijector_Info]] = None,
    epochs: int = 100,
    batch_size: int = 1024,
    optimizer: Callable = None,
    patience: int = 10,
    seed: int = 0,
    progress: bool = True,
) -> list:
    """Trains the normalizing flow on the provided inputs."""
    # split the seed
    root_key = random.PRNGKey(seed)
    init_key, batch_key, bijector_key = random.split(root_key, num=3)

    # if no loss_fn is provided, use the default loss function

    @jit
    def loss_fn(params, x, c):
        return -jnp.mean(flow.log_prob(params, x, c))

    params = flow.init(X_train, C_train, init_key, bijector)

    # initialize the optimizer
    optimizer = optax.adam(learning_rate=1e-3) if optimizer is None else optimizer
    opt_state = optimizer.init(params)

    # define the training step function
    @jit
    def step(params, opt_state, x, c):
        gradients = value_and_grad(loss_fn)(params, x, c)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # save the initial loss
    losses = [loss_fn(params, X_train, C_train).item()]
    test_losses = []

    if progress:
        from rich.progress_bar import track

        loop = track(range(epochs))
    else:
        loop = range(epochs)

    # initialize variables for early stopping
    best_loss = jnp.inf
    best_params = params

    for epoch in loop:
        # new permutation of batches
        permute_key, sample_key, batch_key = random.split(batch_key, num=3)

        perm = random.permutation(permute_key, X_train.shape[0])
        X = X_train[perm]
        C = C_train[perm]

        # loop through batches and step optimizer
        for batch_idx in range(0, len(X), batch_size):
            Xb = X[batch_idx : batch_idx + batch_size]
            Cb = C[batch_idx : batch_idx + batch_size]
            params, opt_state = step(params, opt_state, Xb, Cb)

        losses.append(loss_fn(params, X, C).item())
        test_losses.append(loss_fn(params, X_test, C_test).item())

        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            best_params = params

        stop = np.isnan(losses[-1]) or (
            len(test_losses) > 2 * patience
            and not np.min(test_losses[-patience:])
            < np.min(test_losses[-2 * patience : -patience])
        )

        if stop:
            break

    return best_params, best_loss
