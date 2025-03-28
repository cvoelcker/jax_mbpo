from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from mbpo.nn.mlp import MLP


class StateValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(
            observations, training=training
        )
        return jnp.squeeze(critic, -1)
