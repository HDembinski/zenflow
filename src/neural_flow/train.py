"""Train flow."""

from .flow import Flow
from .typing import Pytree, Array
import jax.numpy as jnp
from typing import Tuple, List, Optional
import numpy as np
import jax
import optax


def train(
    flow: Flow,
    X_train: Array,
    X_test: Array,
    C_train: Optional[Array] = None,
    C_test: Optional[Array] = None,
    epochs: int = 100,
    batch_size: int = 1024,
    optimizer: optax.GradientTransformation = optax.nadamw(learning_rate=1e-3),
    patience: int = 10,
    seed: int = 0,
    progress: bool = True,
) -> Tuple[Pytree, int, List[float], List[float]]:
    """Trains the normalizing flow on the provided inputs."""
    root_key = jax.random.PRNGKey(seed)
    init_key, iter_key = jax.random.split(root_key)

    X_train = jax.device_put(X_train)
    X_test = jax.device_put(X_test)
    if C_train is not None:
        C_train = jax.device_put(C_train)
    if C_test is not None:
        C_test = jax.device_put(C_test)

    variables = flow.init(init_key, X_train, C_train)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch_stats, x, c, train):
        lp, updates = flow(
            {"params": params, "batch_stats": batch_stats},
            x,
            c,
            train=train,
            mutable=["batch_stats"],
        )
        return -jnp.mean(lp).item(), updates

    @jax.jit
    def step(params, batch_stats, opt_state, x, c):
        gradients, updates = jax.grad(loss_fn, with_aux=True)(
            params, batch_stats, x, c, True
        )
        batch_stats = updates["batch_stats"]
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, batch_stats, opt_state

    loss_train = []
    loss_test = []

    if progress:
        from rich.progress import track

        loop = track(range(epochs))
    else:
        loop = range(epochs)

    best_epoch = 0
    best_params = variables
    for epoch in loop:
        permute_key = jax.random.fold_in(iter_key, epoch)
        perm = jax.random.permutation(permute_key, X_train.shape[0])
        X_perm = X_train[perm]
        if C_train is not None:
            C_perm = C_train[perm]

        # loop through batches and step optimizer
        for batch_idx in range(0, len(X_perm), batch_size):
            X = X_perm[batch_idx : batch_idx + batch_size]
            if C_train is not None:
                C = C_perm[batch_idx : batch_idx + batch_size]
            else:
                C = None
            params, batch_stats, opt_state = step(params, batch_stats, opt_state, X, C)

        loss_train.append(loss_fn(params, batch_stats, X, C, False)[0])
        loss_test.append(loss_fn(params, batch_stats, X_test, C_test, False)[0])

        if loss_test[-1] < loss_test[best_epoch]:
            best_epoch = epoch
            best_params = {"params": params, "batch_stats": batch_stats}

        stop = np.isnan(loss_train[-1])

        if epoch >= 2 * patience and epoch % patience == 0:
            stop |= not np.min(loss_test[-patience:]) < np.min(
                loss_test[-2 * patience : -patience]
            )

        if stop:
            break

    return best_params, best_epoch, loss_train, loss_test
