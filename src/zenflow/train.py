"""Train flow."""

from .flow import Flow
from flax.typing import ArrayPytree, Array
import jax.numpy as jnp
from typing import Tuple, List, Optional
import numpy as np
import jax
import optax
import warnings

if hasattr(optax, "nadamw"):
    DEFAULT_OPTIMIZER = optax.nadamw
else:
    DEFAULT_OPTIMIZER = optax.adamw


def train(
    flow: Flow,
    X_train: Array,
    X_test: Array,
    C_train: Optional[Array] = None,
    C_test: Optional[Array] = None,
    *,
    epochs: int = 1000,
    batch_size: int = 1024,
    optimizer: optax.GradientTransformation = DEFAULT_OPTIMIZER(learning_rate=1e-3),
    patience: float = 0.05,
    warmup: float = 0.2,
    seed: int = 0,
    progress: bool = True,
    initial_variables: Optional[ArrayPytree] = None,
) -> Tuple[ArrayPytree, int, List[float], List[float]]:
    """Trains the normalizing flow on the provided inputs."""
    if warmup < 1:
        warmup = warmup * epochs
    warmup = int(warmup)

    if patience < 1:
        patience = patience * epochs
    patience = int(patience)

    X_train = jax.device_put(X_train)
    X_test = jax.device_put(X_test)
    if C_train is not None:
        C_train = jax.device_put(C_train)
    if C_test is not None:
        C_test = jax.device_put(C_test)

    root_key = jax.random.PRNGKey(seed)
    init_key, iter_key = jax.random.split(root_key)

    if initial_variables is None:
        variables = flow.init(
            init_key, X_train[:1], None if C_train is None else C_train[:1]
        )
    else:
        variables = initial_variables
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch_stats, x, c):
        lp, updates = flow.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            c,
            train=True,
            mutable=["batch_stats"],
        )
        return -jnp.mean(lp), updates

    @jax.jit
    def metric_fn(variables, x, c):
        lp = flow.apply(variables, x, c)
        return -jnp.mean(lp)

    @jax.jit
    def step(params, batch_stats, opt_state, x, c):
        gradients, updates = jax.grad(loss_fn, has_aux=True)(params, batch_stats, x, c)
        batch_stats = updates["batch_stats"]
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, batch_stats, opt_state

    loss_train = []
    loss_test = []

    if progress:
        try:
            from tqdm.notebook import tqdm as track
        except ModuleNotFoundError:
            from rich.progress import track

        loop = track(range(epochs))
    else:
        loop = range(epochs)

    best_epoch = 0
    best_variables = variables
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

        variables = {"params": params, "batch_stats": batch_stats}
        loss_train.append(metric_fn(variables, X, C).item())
        loss_test.append(metric_fn(variables, X_test, C_test).item())

        if not np.isfinite(loss_train[-1]):
            msg = f"epoch {epoch}: loss[train] not finite, abort training"
            warnings.warn(msg, RuntimeWarning)
            break

        if loss_test[-1] <= loss_test[best_epoch]:
            best_epoch = epoch
            best_variables = variables

        if epoch >= warmup and epoch >= 2 * patience and epoch % patience == 0:
            if not np.min(loss_test[-patience:]) < np.min(
                loss_test[-2 * patience : -patience]
            ):
                break

    return best_variables, best_epoch, loss_train, loss_test
