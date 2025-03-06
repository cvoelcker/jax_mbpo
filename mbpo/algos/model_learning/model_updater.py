from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from mbpo.data.dataset import DatasetDict
from mbpo.types import Params


def update_model(
    model: TrainState,
    batch: DatasetDict,
) -> Tuple[TrainState, Dict[str, float]]:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"][..., jnp.newaxis]
    next_state = jnp.concatenate([next_state, reward], axis=-1)
    mask = batch["masks"]

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pred_state_dist = model.apply_fn({"params": model_params}, state, action)
        state_loss = -(
            mask[:, jnp.newaxis]
            * pred_state_dist.log_prob(next_state[:, jnp.newaxis, :])
        ).mean()

        state_pred = pred_state_dist.mean()[:, :, :-1]
        state_acc = jnp.mean((state_pred - next_state[:, jnp.newaxis, :-1]) ** 2)

        reward_pred = pred_state_dist.mean()[:, :, -1:]
        reward_acc = jnp.mean((reward_pred - reward[:, jnp.newaxis, :]) ** 2)

        return state_loss, {
            "state_loss": state_loss,
            "reward_acc": reward_acc,
            "state_acc": state_acc,
            "state_std": jnp.mean(pred_state_dist.stddev()[:, :, :-1]),
            "reward_std": jnp.mean(pred_state_dist.stddev()[:, :, -1:]),
        }

    grads, info = jax.grad(model_loss_fn, has_aux=True)(model.params)
    new_model = model.apply_gradients(grads=grads)
    return new_model, info


def per_ensemble_loss(
    model: TrainState,
    batch: DatasetDict,
) -> jax.Array:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"][..., jnp.newaxis]
    mask = batch["masks"]
    next_state = jnp.concatenate([next_state, reward], axis=-1)
    pred_state_dist = model.apply_fn({"params": model.params}, state, action)
    state_loss = jnp.mean(
        mask[:, jnp.newaxis] * pred_state_dist.log_prob(next_state[:, jnp.newaxis, :]),
        (0),
    )
    return state_loss
