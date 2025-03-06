from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from mbpo.nn.mlp import MLP


class GaussianEnsembleModel(nn.Module):
    hidden_dims: Sequence[int]
    num_ensemble: int
    output_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False,
    ) -> distrax.Distribution:
        state = jnp.concatenate([observations, action], axis=-1)
        if len(state.shape) < 2 or state.shape[-2] != self.num_ensemble:
            state = jnp.expand_dims(state, axis=-2).repeat(self.num_ensemble, axis=-2)
        outputs = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(-2, None),
            out_axes=-2,
            axis_size=self.num_ensemble,
        )(self.hidden_dims, activate_final=False, dropout_rate=self.dropout_rate)(
            state, training
        )

        means_and_rewards = nn.Dense(self.output_dim + 1)(outputs)

        log_stds = nn.Dense(self.output_dim + 1)(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return distrax.MultivariateNormalDiag(
            loc=means_and_rewards, scale_diag=jnp.exp(log_stds)
        )
