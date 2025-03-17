"""Implementations of algorithms for continuous control."""

import functools
from typing import Dict, Sequence, Tuple, Callable, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.core import frozen_dict
from flax.training.train_state import TrainState

from mbpo.algos.agent import Agent
from mbpo.algos.model_learning.model_updater import (
    update_nll,
    per_ensemble_nll,
    per_ensemble_vagram,
    per_ensemble_vaml,
    update_vaml,
    update_vagram,
)
from mbpo.algos.sac.sac_learner import SACLearner
from mbpo.data.dataset import Dataset, DatasetDict
from mbpo.nn.model.deterministic_env_model import DeterministicEnsembleModel
from mbpo.nn.model.gaussian_env_model import GaussianEnsembleModel


@functools.partial(jax.jit, static_argnames=("mode"))
def _update_jit(
    model: TrainState,
    batch: DatasetDict,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    critic: TrainState,
    actor: TrainState,
    rng: jax.Array,
    mode: str = "nll",
) -> Tuple[TrainState, Dict[str, float]]:
    update_rng, rng = jax.random.split(rng)
    # update mean and std
    state = batch["observations"]
    action = batch["actions"]
    model_inp = jnp.concatenate([state, action], axis=-1)
    obs_mean = jnp.mean(model_inp, axis=0) * 0.0001 + obs_mean * 0.9999
    obs_std = jnp.std(model_inp, axis=0) * 0.0001 + obs_std * 0.9999
    if mode == "nll":
        return *update_nll(model, batch, obs_mean, obs_std), obs_mean, obs_std, update_rng
    elif mode == "vagram":
        return *update_vagram(model, batch, obs_mean, obs_std, critic, actor, rng), obs_mean, obs_std, update_rng
    elif mode == "vaml":
        return *update_vaml(model, batch, obs_mean, obs_std, critic, actor, rng), obs_mean, obs_std, update_rng
    else:
        raise ValueError(f"Unknown mode: {mode}")


@functools.partial(jax.jit, static_argnames=("mode"))
def compute_loss(
    model: TrainState,
    batch: FrozenDict,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    critic: TrainState,
    actor: TrainState,
    rng: jax.Array,
    mode: str = "nll",
) -> tuple[jax.Array, jax.Array]:
    if mode == "nll":
        return per_ensemble_nll(model, batch, obs_mean, obs_std).mean()
    elif mode == "vagram":
        return per_ensemble_vagram(model, batch, obs_mean, obs_std, critic, actor, rng).mean()
    elif mode == "vaml":
        return per_ensemble_vaml(model, batch, obs_mean, obs_std, critic, actor, rng).mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")


@functools.partial(jax.jit, static_argnames=("mode"))
def compute_per_ensemble_loss(
    model: TrainState, batch: FrozenDict, obs_mean: jax.Array, obs_std: jax.Array, critic: TrainState, actor: TrainState, rng: jax.Array, mode: str = "nll"
) -> tuple[jax.Array, jax.Array]:
    if mode == "nll":
        return per_ensemble_nll(model, batch, obs_mean, obs_std)
    elif mode == "vagram":
        return per_ensemble_vagram(model, batch, obs_mean, obs_std, critic, actor, rng)
    elif mode == "vaml":
        return per_ensemble_vaml(model, batch, obs_mean, obs_std, critic, actor, rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _concatenate_pytree(pytree1, pytree2, length, ratio):
    idx = int(length * ratio)
    return jax.tree_map(lambda x, y: jnp.concatenate([x[:idx], y[idx:]], axis=0), pytree1, pytree2)


@functools.partial(jax.jit, static_argnames=("depth", "terminal_fn", "prop_real"))
def compute_model_based_batch(
    rng: jax.Array,
    model: TrainState,
    policy: TrainState,
    batch: FrozenDict,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    depth: int,
    terminal_fn: Callable[[jax.Array], jax.Array],
    elites: jax.Array,
    prop_real: float = 0.0
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

    for i in range(depth):
        (
            rng,
            action_rng,
            state_rng,
            elite_rng,
        ) = jax.random.split(rng, 4)
        action_dist = policy.apply_fn({"params": policy.params}, state)
        action = action_dist.sample(seed=action_rng)
        state_ensemble_dist = model.apply_fn(
            {"params": model.params}, state, action, obs_mean, obs_std
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
        done = terminal_fn(state)[..., jnp.newaxis]
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
    dones = jnp.clip(jnp.cumsum(dones, axis=0), 0.0, 1.0)

    idxs = jax.random.randint(rng, (1, bs, 1), 0, len(states) - depth)
    states = jnp.take_along_axis(states, idxs, axis=0).squeeze(0)
    actions = jnp.take_along_axis(actions, idxs, axis=0).squeeze(0)
    next_states = jnp.take_along_axis(next_states, idxs, axis=0).squeeze(0)
    rewards = jnp.take_along_axis(rewards, idxs, axis=0).squeeze(0)
    dones = jnp.take_along_axis(dones, idxs, axis=0).squeeze()

    new_batch = frozen_dict.freeze(
        dict(
            observations=states,
            actions=actions,
            next_observations=next_states,
            rewards=rewards.squeeze(),
            masks=1.0 - dones,
            dones=dones,
        )
    )
    return _concatenate_pytree(batch, new_batch, bs, prop_real), rng


class ModelLearner(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_lr: float = 3e-4,
        model_hidden_dims: Sequence[int] = (256, 256),
        model_weight_norm: bool = False,
        n_ensemble: int = 8,
        n_elites: int = 5,
        patience: int = 10,
        loss_mode: str = "nll",
        deterministic: bool = False,
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

        if deterministic:
            model_def = DeterministicEnsembleModel(
                model_hidden_dims,
                num_ensemble=n_ensemble,
                output_dim=observations.shape[-1],
            )
        else:
            model_def = GaussianEnsembleModel(
                model_hidden_dims,
                num_ensemble=n_ensemble,
                output_dim=observations.shape[-1],
            )
        # normalization stats
        dummy_inp = jnp.concatenate([observations, actions], axis=-1)
        self._obs_mean = jnp.zeros_like(dummy_inp)
        self._obs_std = jnp.ones_like(dummy_inp)
        model_params = model_def.init(
            model_key, observations, actions, self._obs_mean, self._obs_std
        )["params"]
        model = TrainState.create(
            apply_fn=model_def.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=model_lr, weight_decay=5e-5),
        )

        self._model = model
        self._models = []
        self._rng = rng
        self._elites = jnp.arange(n_elites)
        self._patience = patience
        self._loss_mode = loss_mode

    def update_step(self, batch: FrozenDict, policy: SACLearner) -> Dict[str, float]:
        critic = policy._critic
        actor = policy._actor
        (
            new_model,
            info,
            obs_mean,
            obs_std,
            self._rng
        ) = _update_jit(
            self._model,
            batch,
            self._obs_mean,
            self._obs_std,
            critic,
            actor,
            self._rng,
            mode=self._loss_mode,
        )

        self._model = new_model
        self._obs_mean = obs_mean
        self._obs_std = obs_std

        return info

    def yield_data(
        self,
        dataset: Dataset,
        policy: SACLearner,
        batch_size: int,
        num_samples: int,
        depth=1,
        termination_fn=lambda x: jnp.zeros_like(x[:, :1]),
        prop_real=0.0,
    ):
        print(prop_real)

        for i, batch in enumerate(looping_epoch_iter(dataset, batch_size)):
            if i == num_samples:
                break
            batch, rng = compute_model_based_batch(
                self._rng,
                self._model,
                policy._actor,
                batch,
                self._obs_mean,
                self._obs_std,
                depth,
                termination_fn,
                self._elites,
                prop_real,
            )
            self._rng = rng
            yield batch

    def compute_elites(self, dataset: Dataset, policy, batch_size: int):
        critic = policy._critic
        actor = policy._actor
        _losses: list[jax.Array] = []
        for batch in dataset.get_epoch_iter(batch_size):
            loss = compute_per_ensemble_loss(
                self._model,
                batch,
                self._obs_mean,
                self._obs_std,
                critic,
                actor,
                self._rng,
                mode=self._loss_mode,
            )
            _losses.append(loss)
        losses = jnp.stack(_losses, axis=0)
        elites = jnp.argsort(losses.mean(0), axis=0, descending=True)[: self._n_elites]
        self._elites = elites
        return losses[self._elites]

    def set_best_model(self, losses):
        idx = jnp.argmax(jnp.array(losses))
        self._model = self._models[idx]
        self._models = []
        return losses[idx]

    def update_model(
        self,
        dataset: Dataset,
        policy: SACLearner,
        batch_size: int,
        max_iters: int | None = None,
    ):
        train_dataset, val_dataset = dataset.split(0.8)

        val_losses = []
        update = True
        iters = 0
        best = jnp.finfo(jnp.float32).min
        best_iter = 0
        while update:
            iters += 1
            batch_infos = []
            for batch in train_dataset.get_epoch_iter(batch_size):
                info = self.update_step(batch, policy)
                batch_infos.append(info)
            self._models.append(self._model)
            epoch_val_loss = 0.0
            val_iters = 0.0
            for batch in val_dataset.get_epoch_iter(batch_size):
                self._rng, rng = jax.jit(jax.random.split)(self._rng)
                _val_loss = compute_loss(
                    self._model, batch, self._obs_mean, self._obs_std, policy._critic, policy._actor, rng, mode=self._loss_mode
                )
                _val_loss = jnp.nan_to_num(_val_loss)
                epoch_val_loss += _val_loss
                val_iters += 1
            epoch_val_loss /= val_iters
            val_losses.append(epoch_val_loss)
            best = jnp.maximum(best, epoch_val_loss)
            best_iter = iters if epoch_val_loss == best else best_iter
            update = (
                len(val_losses) < self._patience or iters - best_iter < self._patience
            )
            if max_iters is not None and iters == max_iters:
                break
        info["iters"] = iters
        info["val_loss"] = epoch_val_loss
        info["best_val_loss"] = self.set_best_model(val_losses)
        info["elite_loss"] = self.compute_elites(val_dataset, policy, batch_size)
        return info


def looping_epoch_iter(dataset, batch_size):
    data_iter = dataset.get_epoch_iter(batch_size)
    while True:
        try:
            yield next(data_iter)
        except StopIteration:
            data_iter = dataset.get_epoch_iter(batch_size)
            yield next(data_iter)
