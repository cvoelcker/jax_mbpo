from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
    size_param: float = 1.0,
):
    "TODO: push to jax"
    return jax.nn.initializers.variance_scaling(
        0.3333 * size_param,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]) -> jnp.ndarray:
    if hasattr(x, "values"):
        return jnp.concatenate([_flatten_dict(v) for k, v in sorted(x.items())], -1)
    else:
        return x


def l2_normalization_activation(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.elu
    activate_final: int = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    add_weight_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)  # , kernel_init=torch_he_uniform())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
            if i + 2 == len(self.hidden_dims) and self.add_weight_norm:
                x = l2_normalization_activation(x)
        return x
