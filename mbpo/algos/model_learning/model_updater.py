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
    mask = batch["masks"][..., jnp.newaxis]

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pred_state_dist, pred_reward = model.apply_fn(
            {"params": model_params}, state, action
        )
        state_loss = -(
            mask[:, jnp.newaxis, :]
            * pred_state_dist.log_prob(next_state[:, jnp.newaxis, :])
        ).mean()
        reward_loss = (
            mask[:, jnp.newaxis, :] * (pred_reward - reward[:, jnp.newaxis, :]) ** 2
        ).mean()
        return state_loss + reward_loss, {
            "state_loss": state_loss,
            "reward_loss": reward_loss,
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
    pred_state_dist, _ = model.apply_fn({"params": model.params}, state, action)
    state_loss = jnp.mean(pred_state_dist.log_prob(next_state[:, jnp.newaxis, :]), (0))
    return state_loss
