"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from mbpo.algos.agent import Agent
from mbpo.algos.sac.actor_update_step import update_actor
from mbpo.algos.sac.critic_update_step import update_critic
from mbpo.algos.sac.temperature import Temperature
from mbpo.algos.sac.temperature_update_step import update_temperature
from mbpo.nn.normal_tanh_policy import NormalTanhPolicy
from mbpo.nn.values import StateActionEnsemble
from mbpo.types import Params, PRNGKey
from mbpo.utils.target_update import soft_target_update


@functools.partial(
    jax.jit, static_argnames=("backup_entropy", "critic_reduction", "update_target", "update_temperature")
)
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    critic_target_params: Params,
    temp: TrainState,
    batch: FrozenDict,
    discount: float,
    tau: float,
    target_entropy: float,
    backup_entropy: bool,
    critic_reduction: str,
    update_target: bool,
    update_temperature: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    rng, key = jax.random.split(rng)
    critic_target = critic.replace(params=critic_target_params)
    new_critic, critic_info = update_critic(
        key,
        actor,
        critic,
        critic_target,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
        critic_reduction=critic_reduction,
    )
    if update_target:
        new_critic_target_params = soft_target_update(
            new_critic.params, critic_target_params, tau
        )
    else:
        new_critic_target_params = critic_target_params

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    if update_temperature:
        new_temp, alpha_info = update_temperature(
            temp, actor_info["entropy"], target_entropy
        )
    else:
        new_temp = temp
        alpha_info = {}

    return (
        rng,
        new_actor,
        new_critic,
        new_critic_target_params,
        new_temp,
        {**critic_info, **actor_info, **alpha_info},
    )


class SACTrainer(Agent):
    def __init__(
        self,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        critic_reduction: str = "min",
        init_temperature: float = 1.0,
        target_update_interval: int = 1,
        critic_weight_norm: bool = False,
        actor_weight_norm: bool = False,
        update_temperature: bool = True,
        **kwargs,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount

        self.update_temperature = update_temperature

        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if np.all(action_space.low == -1) and np.all(action_space.high == 1):
            low = None
            high = None
        else:
            low = action_space.low
            high = action_space.high

        actor_def = NormalTanhPolicy(
            hidden_dims,
            action_dim,
            low=low,
            high=high,
            add_weight_norm=actor_weight_norm,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_def = StateActionEnsemble(
            hidden_dims, num_qs=2, add_weight_norm=critic_weight_norm
        )
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        critic_target_params = copy.deepcopy(critic_params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        self.target_update_interval = target_update_interval

        self._actor = actor
        self._critic = critic
        self._critic_target_params = critic_target_params
        self._temp = temp
        self._rng = rng

    def update(self, batch: FrozenDict, step) -> Dict[str, float]:
        (
            new_rng,
            new_actor,
            new_critic,
            new_critic_target_params,
            new_temp,
            info,
        ) = _update_jit(
            self._rng,
            self._actor,
            self._critic,
            self._critic_target_params,
            self._temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.backup_entropy,
            self.critic_reduction,
            step % self.target_update_interval == 0,
            self.update_temperature,
        )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._critic_target_params = new_critic_target_params
        self._temp = new_temp

        return info

    def get_checkpoint(self):
        return {
            "actor": self._actor,
            "critic": self._critic,
            "critic_target_params": self._critic_target_params,
            "temp": self._temp,
        }

    def load_checkpoint(self, checkpoint):
        self._actor = checkpoint["actor"]
        self._critic = checkpoint["critic"]
        self._critic_target_params = checkpoint["critic_target_params"]
        self._temp = checkpoint["temp"]
