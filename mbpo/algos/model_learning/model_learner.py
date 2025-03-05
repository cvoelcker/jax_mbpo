"""Implementations of algorithms for continuous control."""

import functools
from typing import Dict, Sequence, Tuple, Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.core import frozen_dict
from flax.training.train_state import TrainState

from mbpo.algos.agent import Agent
from mbpo.algos.model_learning.model_updater import update_model, per_ensemble_loss
from mbpo.algos.sac.sac_learner import SACLearner
from mbpo.data.dataset import Dataset, DatasetDict
from mbpo.nn.model.gaussian_env_model import GaussianEnsembleModel
from mbpo.types import Params, PRNGKey


@jax.jit
def _update_jit(
    model: TrainState,
    batch: DatasetDict,
) -> Tuple[TrainState, Dict[str, float]]:
    return update_model(model, batch)

@jax.jit
def compute_loss(
    model: TrainState, batch: FrozenDict
) -> tuple[jax.Array, jax.Array]:
    return per_ensemble_loss(model, batch).mean()

@jax.jit
def compute_per_ensemble_loss(
    model: TrainState, batch: FrozenDict
) -> tuple[jax.Array, jax.Array]:
    return per_ensemble_loss(model, batch)


@functools.partial(jax.jit, static_argnames=("depth", "terminal_fn"))
def compute_model_based_batch(
    rng: jax.Array,
    model: TrainState,
    policy: TrainState,
    batch: FrozenDict,
    depth: int,
    terminal_fn: Callable[[jax.Array], jax.Array],
    elites: jax.Array,
) -> frozen_dict.FrozenDict:
    n_elites = elites.shape[0]
    bs = batch["observations"].shape[0]

    _states = []
    _actions = []
    _next_states = []
    _rewards = []
    _dones = []

    state = batch["observations"]

    _states.append(state)

    for _ in range(depth):
        (
            rng,
            action_rng,
            state_rng,
            elite_rng,
        ) = jax.random.split(rng, 4)
        action_dist = policy.apply_fn({"params": policy.params}, state)
        action = action_dist.sample(seed=action_rng)
        state_ensemble_dist  = model.apply_fn(
            {"params": model.params}, state, action
        )
        state_ensemble = state_ensemble_dist.sample(seed=state_rng)
        _elite_idx = jax.random.randint(elite_rng, (bs,), 0, n_elites)
        elite_idxs = elites[_elite_idx]
        state = jnp.take_along_axis(
            state_ensemble[..., :-1], elite_idxs[:, None, None], axis=-2
        ).squeeze(-2)
        reward = jnp.take_along_axis(
            state_ensemble[..., -1:], elite_idxs[:, None, None], axis=-2
        ).squeeze(-2)
        done = terminal_fn(state)
        _states.append(state)
        _actions.append(action)
        _rewards.append(reward)
        _dones.append(done)
        _next_states.append(state)

    states = jnp.stack(_states, axis=0)
    actions = jnp.stack(_actions, axis=0)
    next_states = jnp.stack(_next_states, axis=0)
    rewards = jnp.stack(_rewards, axis=0)
    dones = jnp.stack(_dones, axis=0)

    idxs = jax.random.randint(rng, (1, bs, 1), 0, len(states) - depth)
    states = jnp.take_along_axis(states, idxs, axis=0).squeeze(0)
    actions = jnp.take_along_axis(actions, idxs, axis=0).squeeze(0)
    next_states = jnp.take_along_axis(next_states, idxs, axis=0).squeeze(0)
    rewards = jnp.take_along_axis(rewards, idxs, axis=0).squeeze(0)
    dones = jnp.take_along_axis(dones, idxs, axis=0).squeeze(0)

    return frozen_dict.freeze(
        dict(
            observations=states,
            actions=actions,
            next_observations=next_states,
            rewards=rewards.squeeze(),
            masks=1.0 - dones.squeeze(),
        )
    ), rng


class ModelLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        n_ensemble: int = 8,
        n_elites: int = 5,
        **kwargs,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self._n_elites = n_elites

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, model_key = jax.random.split(rng)

        model_def = GaussianEnsembleModel(
            hidden_dims, num_ensemble=n_ensemble, output_dim=observations.shape[-1]
        )
        actor_params = model_def.init(model_key, observations, actions)["params"]
        model = TrainState.create(
            apply_fn=model_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=model_lr),
        )

        self._model = model
        self._rng = rng
        self._elites = jnp.arange(n_elites)

    def update_step(self, batch: FrozenDict) -> Dict[str, float]:
        (
            new_model,
            info,
        ) = _update_jit(
            self._model,
            batch,
        )

        self._model = new_model

        return info

    def yield_data(
        self, dataset: Dataset, policy: SACLearner, batch_size: int, num_samples: int, depth = 1, terminal_fn= lambda x: jnp.zeros_like(x[:, :1])
    ):
        for _ in range(num_samples):
            batch = dataset.sample(batch_size)
            batch, rng = compute_model_based_batch(
                self._rng, self._model, policy._actor, batch, depth, terminal_fn, self._elites
            )
            self._rng = rng
            yield batch

    def compute_elites(self, dataset: Dataset, batch_size: int):
        _losses: list[jax.Array] = []
        for batch in dataset.get_epoch_iter(batch_size):
            loss = compute_per_ensemble_loss(self._model, batch)
            _losses.append(loss)
        losses = jnp.stack(_losses, axis=0)
        elites = jnp.argsort(losses.mean(0), axis=0)[: self._n_elites]
        self._elites = elites

    def update_model(self, dataset: Dataset, batch_size: int, max_iters: int | None = 200):
        train_dataset, val_dataset = dataset.split(0.8)
        val_losses = []
        update = True
        iters = 0
        while update:
            iters += 1
            batch_infos = []
            for batch in train_dataset.get_epoch_iter(batch_size):
                info = self.update_step(batch)
                batch_infos.append(info)
            epoch_val_loss = 0.0
            for batch in val_dataset.get_epoch_iter(batch_size):
                epoch_val_loss += compute_loss(self._model, batch)
            val_losses.append(epoch_val_loss)
            update = (
                len(val_losses) < 10
                or val_losses[-1] > jnp.array(val_losses[-10:]).mean()
            )
            print(f"Epoch {iters} val loss: {epoch_val_loss}")
            if max_iters is not None and iters == max_iters:
                break 
        info["iters"] = iters
        self.compute_elites(val_dataset, batch_size)
        return info, val_losses
