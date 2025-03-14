from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]) -> jnp.ndarray:
    if hasattr(x, "values"):
        return jnp.concatenate([_flatten_dict(v) for k, v in sorted(x.items())], -1)
    else:
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                # x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x
