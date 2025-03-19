from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from mbpo.data.dataset import DatasetDict
from mbpo.types import Params


def update_nll(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
) -> Tuple[TrainState, Dict[str, float]]:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"]
    target_state = jnp.concatenate([next_state, reward[..., jnp.newaxis]], axis=-1)

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pred_state_dist = model.apply_fn(
            {"params": model_params}, state, action, obs_mean, obs_std
        )
        state_loss = (
            -(pred_state_dist.log_prob(target_state[:, jnp.newaxis, :]))
            .sum(axis=-1)
            .mean()
            / state.shape[-1]
        )

        state_pred = pred_state_dist.mean()[:, :, :-1]
        state_acc = jnp.mean((state_pred - target_state[:, jnp.newaxis, :-1]) ** 2)

        reward_pred = pred_state_dist.mean()[:, :, -1]
        reward_acc = jnp.mean((reward_pred - reward[:, jnp.newaxis]) ** 2)

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


def per_ensemble_nll(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jax.Array,
    obs_std: jax.Array,
) -> jax.Array:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"]
    target_state = jnp.concatenate([next_state, reward[..., jnp.newaxis]], axis=-1)

    pred_state_dist = model.apply_fn(
        {"params": model.params}, state, action, obs_mean, obs_std
    )
    state_loss = jnp.mean(
        pred_state_dist.log_prob(target_state[:, jnp.newaxis, :]),
        (0),
    )
    return state_loss


def update_vaml(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    critic: TrainState,
    actor: TrainState,
    rng: jax.Array,
) -> Tuple[TrainState, Dict[str, float]]:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"]
    target_state = jnp.concatenate([next_state, reward[..., jnp.newaxis]], axis=-1)

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pred_state_dist = model.apply_fn(
            {"params": model_params}, state, action, obs_mean, obs_std
        )

        pred_state = pred_state_dist.sample(seed=rng)[..., :-1]
        next_action = jax.lax.stop_gradient(
            actor.apply_fn({"params": actor.params}, pred_state)
        )
        real_next_action = jax.lax.stop_gradient(
            actor.apply_fn({"params": actor.params}, next_state)
        )

        # compute all 4 Q values
        q1 = jax.lax.stop_gradient(
            critic.apply_fn({"params": critic.params}, next_state, next_action)
        )
        q2 = jax.lax.stop_gradient(
            critic.apply_fn({"params": critic.params}, next_state, real_next_action)
        )
        q1_target = critic.apply_fn({"params": critic.target}, pred_state, action)
        q2_target = critic.apply_fn(
            {"params": critic.target}, pred_state, real_next_action
        )

        # compute loss
        q1_loss = jnp.mean((q1 - q1_target) ** 2)
        q2_loss = jnp.mean((q2 - q2_target) ** 2)

        state_loss = q1_loss + q2_loss

        # compute logging info
        state_pred = pred_state_dist.mean()[:, :, :-1]
        state_acc = jnp.mean((state_pred - target_state[:, jnp.newaxis, :-1]) ** 2)
        reward_pred = pred_state_dist.mean()[:, :, -1]
        reward_acc = jnp.mean((reward_pred - reward[:, jnp.newaxis]) ** 2)

        return state_loss + reward_acc, {
            "state_loss": state_loss,
            "reward_acc": reward_acc,
            "state_acc": state_acc,
            "state_std": jnp.mean(pred_state_dist.stddev()[:, :, :-1]),
            "reward_std": jnp.mean(pred_state_dist.stddev()[:, :, -1:]),
        }

    grads, info = jax.grad(model_loss_fn, has_aux=True)(model.params)
    new_model = model.apply_gradients(grads=grads)
    return new_model, info


def per_ensemble_vaml(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    critic: TrainState,
    actor: TrainState,
    rng: jax.Array,
) -> jax.Array:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"]

    pred_state_dist = model.apply_fn(
        {"params": model.params}, state, action, obs_mean, obs_std
    )

    pred_state = pred_state_dist.sample(seed=rng)[..., :-1]
    next_action = jax.lax.stop_gradient(
        actor.apply_fn({"params": actor.params}, pred_state)
    )
    real_next_action = jax.lax.stop_gradient(
        actor.apply_fn({"params": actor.params}, next_state)
    )

    # compute all 4 Q values
    q1 = jax.lax.stop_gradient(
        critic.apply_fn({"params": critic.params}, next_state, next_action)
    )
    q2 = jax.lax.stop_gradient(
        critic.apply_fn({"params": critic.params}, next_state, real_next_action)
    )
    q1_target = critic.apply_fn({"params": critic.target}, pred_state, action)
    q2_target = critic.apply_fn({"params": critic.target}, pred_state, real_next_action)

    # compute loss
    q1_loss = jnp.mean((q1 - q1_target) ** 2)
    q2_loss = jnp.mean((q2 - q2_target) ** 2)

    state_loss = q1_loss + q2_loss
    reward_acc = jnp.mean(
        (pred_state_dist.sample(seed=rng)[..., -1] - reward[:, jnp.newaxis]) ** 2
    )

    return -(state_loss + reward_acc)


def update_vagram(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    critic: TrainState,
    critic_target: TrainState,
    actor: TrainState,
    rng: jax.Array,
) -> Tuple[TrainState, Dict[str, float]]:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"]
    target_state = jnp.concatenate([next_state, reward[..., jnp.newaxis]], axis=-1)

    value_action = actor.apply_fn({"params": actor.params}, next_state).sample(seed=rng)
    q_sensitivity = jax.vmap(
        jax.jacrev(
            lambda x, a: critic.apply_fn(
                {"params": critic.params}, x.reshape(1, -1), a.reshape(1, -1)
            )
        )
    )(next_state, value_action)
    target_q_sensitivity = jax.vmap(
        jax.jacrev(
            lambda x, a: critic_target.apply_fn(
                {"params": critic_target.params}, x.reshape(1, -1), a.reshape(1, -1)
            )
        )
    )(next_state, value_action)
    target_q_sensitivity = jnp.concatenate(
        [target_q_sensitivity, q_sensitivity], axis=1
    )
    # clamp huge outliers in grad norm
    target_q_sensitivity = clip_by_norm_percentile(target_q_sensitivity, 0.05, 95)
    # make sure we fit the reward accurately
    reward_sensitivity = jnp.repeat(
        jnp.ones_like(reward[:, jnp.newaxis, jnp.newaxis]), 4, 1
    )  * jnp.mean(target_q_sensitivity)
    target_q_sensitivity = jnp.concatenate(
        [target_q_sensitivity.squeeze(-2), reward_sensitivity], axis=-1
    )

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        pred_state_dist = model.apply_fn(
            {"params": model_params}, state, action, obs_mean, obs_std
        )
        state_loss = (
            -(
                jax.vmap(pred_state_dist.weighted_log_prob, in_axes=(None, 1))(
                    target_state[:, jnp.newaxis, :],
                    target_q_sensitivity[:, :, jnp.newaxis, :],
                )
            )
            # .sum((1, 2))
            .mean()
        )
        state_pred = pred_state_dist.mean()[:, :, :-1]
        state_acc = jnp.mean((state_pred - target_state[:, jnp.newaxis, :-1]) ** 2)

        reward_pred = pred_state_dist.mean()[:, :, -1]
        reward_acc = jnp.mean((reward_pred - reward[:, jnp.newaxis]) ** 2)

        return state_loss, {
            "state_loss": state_loss,
            "reward_acc": reward_acc,
            "state_acc": state_acc,
            "state_std": jnp.mean(pred_state_dist.stddev()[:, :, :-1]),
            "reward_std": jnp.mean(pred_state_dist.stddev()[:, :, -1:]),
            "sensitivity_mean": jnp.mean(target_q_sensitivity),
            "sentitivity_min": jnp.min(target_q_sensitivity),
            "sensitivity_max": jnp.max(target_q_sensitivity),
        }

    grads, info = jax.grad(model_loss_fn, has_aux=True)(model.params)
    new_model = model.apply_gradients(grads=grads)
    return new_model, info


def per_ensemble_vagram(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    critic: TrainState,
    critic_target: TrainState,
    actor: TrainState,
    rng: jax.Array,
) -> jax.Array:
    state = batch["observations"]
    action = batch["actions"]
    next_state = batch["next_observations"]
    reward = batch["rewards"]
    target_state = jnp.concatenate([next_state, reward[..., jnp.newaxis]], axis=-1)

    value_action = actor.apply_fn({"params": actor.params}, next_state).sample(seed=rng)
    q_sensitivity = jax.vmap(
        jax.jacrev(
            lambda x, a: critic.apply_fn(
                {"params": critic.params}, x.reshape(1, -1), a.reshape(1, -1)
            )
        )
    )(next_state, value_action)
    target_q_sensitivity = jax.vmap(
        jax.jacrev(
            lambda x, a: critic_target.apply_fn(
                {"params": critic_target.params}, x.reshape(1, -1), a.reshape(1, -1)
            )
        )
    )(next_state, value_action)
    target_q_sensitivity = jnp.concatenate(
        [target_q_sensitivity, q_sensitivity], axis=1
    )
    # clamp huge outliers in grad norm
    target_q_sensitivity = clip_by_norm_percentile(target_q_sensitivity, 0.05, 95)
    # make sure we fit the reward accurately
    reward_sensitivity = jnp.repeat(
        jnp.ones_like(reward[:, jnp.newaxis, jnp.newaxis]), 4, 1
    ) * jnp.mean(target_q_sensitivity)
    target_q_sensitivity = jnp.concatenate(
        [target_q_sensitivity.squeeze(-2), reward_sensitivity], axis=-1
    )

    pred_state_dist = model.apply_fn(
        {"params": model.params}, state, action, obs_mean, obs_std
    )
    state_loss = (
        (
            jax.vmap(pred_state_dist.weighted_log_prob, in_axes=(None, 1))(
                target_state[:, jnp.newaxis, :],
                target_q_sensitivity[:, :, jnp.newaxis, :],
            )
        )
        # .sum(1)
        .mean((0, 1))
    )

    return state_loss


def clip_by_norm_percentile(x, lower, upper):
    norms = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6
    lower_percentile, upper_percentile = jnp.percentile(
        norms.squeeze(), jnp.array([lower, upper])
    )
    clipped_norms = jnp.clip(norms, lower_percentile, upper_percentile)
    return jnp.abs(x * (clipped_norms / norms))
