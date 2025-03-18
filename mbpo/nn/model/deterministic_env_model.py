from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from mbpo.nn.mlp import MLP


class MSEDeterministic(distrax.Deterministic):

    def log_prob(self, x: jax.Array) -> jax.Array:
        return -jnp.square(self.loc - x).mean(axis=-1)

    def weighted_log_prob(self, x: jax.Array, weights: jax.Array) -> jax.Array:
        return -(jnp.square((self.loc - x) * weights)).mean(axis=-1)


class DeterministicEnsembleModel(nn.Module):
    hidden_dims: Sequence[int]
    num_ensemble: int
    output_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None
    add_weight_norm: bool = False

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        action: jnp.ndarray,
        mean: jnp.ndarray,
        std: jnp.ndarray,
        training: bool = False,
    ) -> distrax.Distribution:
        state = jnp.concatenate([observations, action], axis=-1)
        if len(state.shape) < 2 or state.shape[-2] != self.num_ensemble:
            state = jnp.expand_dims(state, axis=-2).repeat(self.num_ensemble, axis=-2)
        state_inp = (state - mean) / (std + 1e-6)
        outputs = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(-2, None),
            out_axes=-2,
            axis_size=self.num_ensemble,
        )(
            self.hidden_dims,
            activations=nn.silu,
            activate_final=False,
            dropout_rate=self.dropout_rate,
            add_weight_norm=self.add_weight_norm,
        )(
            state_inp, training
        )

        means_and_rewards = nn.Dense(self.output_dim + 1)(outputs)

        offset_state = jnp.concatenate(
            [observations, jnp.zeros_like(observations[..., :1])], axis=-1
        )
        if len(offset_state.shape) < 2 or offset_state.shape[-2] != self.num_ensemble:
            offset_state = jnp.expand_dims(offset_state, axis=-2).repeat(self.num_ensemble, axis=-2)

        means_and_rewards = means_and_rewards + offset_state

        return MSEDeterministic(loc=means_and_rewards)
